__version__ = "0.2"

import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class ImageSegmentationPromptNodeSize(desc.MultiDynamicNodeSize):
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

class ImageSegmentationPrompt(desc.Node):
    size = ImageSegmentationPromptNodeSize(['input', 'extensionIn'])
    gpu = desc.Level.INTENSIVE
    parallelization = desc.Parallelization(blockSize=50)

    category = "Utils"
    documentation = """
Generate a binary mask corresponding to the input text prompt.
First a recognition model (image to tags) is launched on the input image.
If the prompt or a synonym is detected in the returned list of tags the detection model (tag to bounded box) is launched.
Detection can be forced by setting to True the appropriate parameter.
If at least one bounded box is returned the segmentation model (bounded box to binary mask) is launched.
Bounded box sizes can be increased by a ratio from 0 to 100%.
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
        desc.File(
            name="detectionModelPath",
            label="Detection Model",
            description="Weights file for the detection model.",
            value="${RDS_DETECTION_MODEL_PATH}",
        ),
        desc.File(
            name="detectionConfigPath",
            label="Detection Config",
            description="Config file for the detection model.",
            value="${RDS_DETECTION_CONFIG_PATH}",
        ),
        desc.File(
            name="segmentationModelPath",
            label="Segmentation Model",
            description="Weights file for the segmentation model.",
            value="${RDS_SEGMENTATION_MODEL_PATH}",
        ),
        desc.StringParam(
            name="prompt",
            label="Prompt",
            description="What to segment, separated by point or one item per line.",
            value="person",
            semantic="multiline",
        ),
        desc.StringParam(
            name="synonyms",
            label="Synonyms",
            description="Synonyms to prompt separated by commas or one item per line. eg: man,woman,boy,girl,human,people can be used as synonyms of person.",
            value="man\nwoman\nboy\ngirl\nhuman\npeople",
            semantic="multiline",
        ),
        desc.BoolParam(
            name="forceDetection",
            label="Force Detection",
            description="Launch detection step even if neither the prompt nor any synonym is recognized.",
            value=False,
        ),
        desc.FloatParam(
            name="thresholdDetection",
            label="Threshold Detection",
            description="Threshold for detection. The lower it is, the more sensitive the detector.",
            range=(0.0, 1.0, 0.1),
            value=0.2,
        ),
        desc.IntParam(
            name="bboxMargin",
            label="Detection Margin",
            description="Increase bounded box dimensions by the selected percentage.",
            range=(0, 100, 1),
            value=0,
        ),
        desc.BoolParam(
            name="maskInvert",
            label="Invert Masks",
            description="Invert mask values. If selected, the pixels corresponding to the mask will be set to 0 instead of 255.",
            value=False,
        ),
        desc.BoolParam(
            name="useGpu",
            label="Use GPU",
            description="Use GPU for computation if available.",
            value=True,
            invalidate=False,
        ),
        desc.BoolParam(
            name="keepFilename",
            label="Keep Filename",
            description="Keep the filename of the inputs for the outputs.",
            value=False,
        ),
        desc.ChoiceParam(
            name="extensionOut",
            label="Output File Extension",
            description="Output image file extension.\n"
                        "If unset, the output file extension will match the input's if possible.",
            value="exr",
            values=["exr", "png", "jpg"],
            exclusive=True,
            group="",  # remove from command line params
        ),
        desc.BoolParam(
            name="outputBboxImage",
            label="Output Bounded Box Image",
            description="Write source image with bounded boxes baked in.",
            value=False,
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
            label="Masks Folder",
            description="Output path for the masks.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="masks",
            label="Masks",
            description="Generated segmentation masks.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + "." + attr.node.extensionOut.value,
            group="",
        ),
        desc.File(
            name="bboxes",
            label="BBoxes",
            description="Generated images with bounded boxes baked in.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + "_bboxes.jpg",
            enabled=lambda node: node.outputBboxImage.value,
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

            processor = segmentation.SegmentationRDS(RAM_CHECKPOINT_PATH = chunk.node.recognitionModelPath.evalValue,
                                                     GD_CONFIG_PATH = chunk.node.detectionConfigPath.evalValue,
                                                     GD_CHECKPOINT_PATH = chunk.node.detectionModelPath.evalValue,
                                                     SAM_CHECKPOINT_PATH = chunk.node.segmentationModelPath.evalValue,
                                                     useGPU = chunk.node.useGpu.value)

            prompt = chunk.node.prompt.value.replace("\n", ".")
            chunk.logger.debug("prompt: {}".format(prompt))
            synonyms = chunk.node.synonyms.value.replace("\n", ",")
            chunk.logger.debug("synonyms: {}".format(synonyms))

            for k, (iFile, oFile) in enumerate(outFiles.items()):
                if k >= chunk.range.start and k <= chunk.range.last:
                    img, h_ori, w_ori, PAR, orientation = image.loadImage(iFile, True)
                    mask, bboxes, conf, tags = processor.process(image = img,
                                                                 prompt = chunk.node.prompt.value,
                                                                 synonyms = chunk.node.synonyms.value,
                                                                 threshold = chunk.node.thresholdDetection.value,
                                                                 force = chunk.node.forceDetection.value,
                                                                 bboxMargin = chunk.node.bboxMargin.value,
                                                                 invert = chunk.node.maskInvert.value,
                                                                 verbose = False)

                    chunk.logger.info("image: {}".format(iFile))
                    chunk.logger.debug("tags: {}".format(tags))
                    chunk.logger.debug("bboxes: {}".format(bboxes))
                    chunk.logger.debug("confidence: {}".format(conf))

                    image.writeImage(oFile[0], mask, h_ori, w_ori, orientation, PAR)

                    if (chunk.node.outputBboxImage.value):
                        imgBoxes = (img * 255.0).astype("uint8")
                        for bbox in bboxes:
                            imgBoxes = image.addRectangle(imgBoxes, bbox)
                        image.writeImage(oFile[1], imgBoxes, h_ori, w_ori, orientation, PAR)

        finally:
            del processor
            torch.cuda.empty_cache()
            chunk.logManager.end()
