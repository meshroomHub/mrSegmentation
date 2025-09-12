__version__ = "0.2"

import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class ImageTagsExtractionNodeSize(desc.MultiDynamicNodeSize):
    def computeSize(self, node):
        from pathlib import Path
        import itertools

        input_path_param = node.attribute(self._params[0])
        extension_param = node.attribute(self._params[1])

        input_path = input_path_param.value
        extension = extension_param.value
        include_suffixes = [extension.lower(), extension.upper()]

        size = 1
        if Path(input_path).is_dir():
            image_paths = list(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
            size = len(image_paths)
        elif node.attribute(self._params[0]).isLink:
            size = node.attribute(self._params[0]).inputLink.node.size
        
        return size

class ImageTagsExtraction(desc.Node):
    size = ImageTagsExtractionNodeSize(['input', 'extensionIn'])
    gpu = desc.Level.INTENSIVE
    parallelization = desc.Parallelization(blockSize=50)

    category = "Utils"
    documentation = """
Generate a set of tags corresponding to recognized elements using a recognition model.
"""

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="Folder or SfMData file.",
            value="",
        ),
        desc.ChoiceParam(
            name="extensionIn",
            label="Input File Extension",
            description="Input image file extension.\n"
                        "Considered only if input is a folder.",
            value="exr",
            values=["exr", "png", "jpg"],
            exclusive=True,
            group="",  # remove from command line params
            enabled=lambda node: Path(node.chunk.input.value).is_dir(),
        ),
        desc.File(
            name="recognitionModelPath",
            label="Recognition Model",
            description="Weights file for the recognition model.",
            value="${RDS_RECOGNITION_MODEL_PATH}",
        ),
        desc.BoolParam(
            name="useGpu",
            label="Use GPU",
            description="Use GPU for computation if available.",
            value=True,
            invalidate=False,
        ),
        desc.BoolParam(
            name="outputTaggedImage",
            label="Output Tagged Image",
            description="Write source image with tags baked in.",
            value=False,
        ),
        desc.BoolParam(
            name="keepFilename",
            label="Keep Filename",
            description="Keep the filename of the inputs for the outputs.",
            value=False,
            enabled=lambda node: node.outputTaggedImage.value,
        ),
        desc.ChoiceParam(
            name="extensionOut",
            label="Output File Extension",
            description="Output image file extension.",
            value="jpg",
            values=["png", "jpg"],
            exclusive=True,
            enabled=lambda node: node.outputTaggedImage.value,
            group="",  # remove from command line params
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug).",
            value="info",
            values=VERBOSE_LEVEL,
            exclusive=True,
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Result Folder",
            description="Output path for the resulting images.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="results",
            label="Results",
            description="Generated images.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + "." + attr.node.extensionOut.value,
            group="",
        ),
    ]

    def resolvedPaths(self, input_path, extensionIn, outDir, keepFilename, extensionOut):
        from pyalicevision import sfmData
        from pyalicevision import sfmDataIO
        from pathlib import Path
        import itertools

        include_suffixes = [extensionIn.lower(), extensionIn.upper()]
        paths = {}
        if Path(input_path).is_dir():
            input_filepaths = sorted(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
            for frameId, inputFile in enumerate(input_filepaths):
                outputFileMask = os.path.join(outDir, Path(inputFile).stem + "." + extensionOut)
                outputFileBoxes = os.path.join(outDir, "bboxes_" + Path(inputFile).stem + ".jpg")
                paths[str(inputFile)] = (outputFileMask, outputFileBoxes, frameId)
        elif Path(input_path).suffix.lower() in [".sfm", ".abc"]:
            if Path(input_path).exists():
                dataAV = sfmData.SfMData()
                if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL) and os.path.isdir(outDir):
                    views = dataAV.getViews()
                    for id, v in views.items():
                        inputFile = v.getImage().getImagePath()
                        frameId = v.getFrameId()
                        if keepFilename:
                            outputFileMask = os.path.join(outDir, Path(inputFile).stem + "." + extensionOut)
                            outputFileBoxes = os.path.join(outDir, "bboxes_" + Path(inputFile).stem + ".jpg")
                        else:
                            outputFileMask = os.path.join(outDir, str(id) + "." + extensionOut)
                            outputFileBoxes = os.path.join(outDir, "bboxes_" + str(id) + ".jpg")
                        paths[inputFile] = (outputFileMask, outputFileBoxes, frameId)

        return paths

    def processChunk(self, chunk):
        import json
        from segmentationRDS import image, segmentation
        import torch

        processor = None
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)

            if not chunk.node.input:
                chunk.logger.warning("Nothing to segment")
                return
            if not chunk.node.output.value:
                return

            chunk.logger.info("Chunk range from {} to {}".format(chunk.range.start, chunk.range.last))

            outFiles = self.resolvedPaths(chunk.node.input.value, chunk.node.extensionIn.value, chunk.node.output.value, chunk.node.keepFilename.value, chunk.node.extensionOut.value)

            if not os.path.exists(chunk.node.output.value):
                os.mkdir(chunk.node.output.value)

            os.environ["TOKENIZERS_PARALLELISM"] = "true"  # required to avoid warning on tokenizers

            processor = segmentation.RecognizeAnything(RAM_CHECKPOINT_PATH = chunk.node.recognitionModelPath.evalValue,
                                                       useGPU = chunk.node.useGpu.value)

            dict = {}

            for k, (iFile, oFile) in enumerate(outFiles.items()):
                if k >= chunk.range.start and k <= chunk.range.last:
                    img, h_ori, w_ori, PAR, orientation = image.loadImage(iFile, True)
                    tags = processor.get_tags(image = img)

                    chunk.logger.info("image: {}".format(iFile))
                    chunk.logger.debug("HxW: {}x{}".format(h_ori, w_ori))
                    chunk.logger.debug("PAR: {}".format(PAR))
                    chunk.logger.debug("orientation: {}".format(orientation))
                    chunk.logger.debug("tags: {}".format(tags))

                    imgInfo = {}
                    imgInfo["tags"] = tags
                    dict[iFile] = imgInfo

                    if (chunk.node.outputTaggedImage.value):
                        imgTags = (img * 255.0).astype("uint8")
                        h,w,c = imgTags.shape
                        txtSize = h // 25
                        for i,tag in enumerate(tags):
                            x = 60 + (i // 12)*(w // 3)
                            y = txtSize + 2*(i%12)*txtSize
                            imgTags = image.addText(imgTags, tag, x, y, txtSize)

                        image.writeImage(oFile[0], imgTags, h_ori, w_ori, orientation, PAR)

            jsonFilename = chunk.node.output.value + "/tags." + str(chunk.index) + ".json"
            jsonObject = json.dumps(dict, indent = 4)
            outfile = open(jsonFilename,"w")
            outfile.write(jsonObject)
            outfile.close()

        finally:
            del processor
            torch.cuda.empty_cache()
            chunk.logManager.end()
