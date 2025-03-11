__version__ = "0.1"

import os

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class ImageBiRefNetMatting(desc.Node):
    size = desc.DynamicNodeSize("input")
    gpu = desc.Level.INTENSIVE
    parallelization = desc.Parallelization(blockSize=50)

    category = "Utils"
    documentation = """
Generate a matte from an input bounded box.
In case no bounding box is provided, the full image is processed.
Based on the BiRefNet model.
"""

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="SfMData file input.",
            value="",
        ),
        desc.File(
            name="bboxFolder",
            label="BBoxes Folder",
            description="JSON file containing prompting bounded boxes.",
            value="",
        ),
        desc.ChoiceParam(
            name="birefnetModelType",
            label="BiRefNet Model Type",
            description="Model type.",
            value='BiRefNet HR Matting',
            values=['BiRefNet LR','BiRefNet HR','BiRefNet HR Matting'],
            exclusive=True,
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
            value=True,
        ),
        desc.ChoiceParam(
            name="extension",
            label="Output File Extension",
            description="Output image file extension.\n"
                        "If unset, the output file extension will match the input's if possible.",
            value="exr",
            values=["exr", "png", "jpg"],
            exclusive=True,
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
            label="Masks Folder",
            description="Output path for the masks.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="masks",
            label="Masks",
            description="Generated segmentation masks.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + "." + attr.node.extension.value,
            group="",
        ),
    ]

    def resolvedPaths(self, inputSfm, outDir, keepFilename, extension):
        import pyalicevision as av
        from pathlib import Path

        paths = {}
        dataAV = av.sfmData.SfMData()
        if av.sfmDataIO.load(dataAV, inputSfm, av.sfmDataIO.ALL) and os.path.isdir(outDir):
            views = dataAV.getViews()
            for id, v in views.items():
                inputFile = v.getImage().getImagePath()
                if keepFilename:
                    outputFileMask = os.path.join(outDir, Path(inputFile).stem + "." + extension)
                    outputFileBoxes = os.path.join(outDir, "bboxes_" + Path(inputFile).stem + ".jpg")
                else:
                    outputFileMask = os.path.join(outDir, str(id) + ".exr")
                    outputFileBoxes = os.path.join(outDir, "bboxes_" + str(id) + "." + extension)
                paths[inputFile] = (outputFileMask, outputFileBoxes)

        return paths

    def processChunk(self, chunk):
        import json
        from segmentationRDS import image, segmentation
        import numpy as np
        import torch

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            processFullImages = False

            if not chunk.node.input:
                chunk.logger.warning("Nothing to segment")
                return
            if not chunk.node.bboxFolder.value:
                chunk.logger.warning("No folder containing bounded boxes, full images will be processed")
                processFullImages = True
            if not chunk.node.output.value:
                return

            chunk.logger.info("Chunk range from {} to {}".format(chunk.range.start, chunk.range.last))

            outFiles = self.resolvedPaths(chunk.node.input.value, chunk.node.output.value, chunk.node.keepFilename.value, chunk.node.extension.value)

            if not os.path.exists(chunk.node.output.value):
                os.mkdir(chunk.node.output.value)

            processor = segmentation.BiRefNetSeg(modelType = chunk.node.birefnetModelType.value,
                                                 useGPU = chunk.node.useGpu.value)
            bboxDict = {}
            if not processFullImages:
                for file in os.listdir(chunk.node.bboxFolder.value):
                    if file.endswith(".json"):
                        with open(os.path.join(chunk.node.bboxFolder.value,file)) as bboxFile:
                            bb = json.load(bboxFile)
                            bboxDict.update(bb)

            for k, (iFile, oFile) in enumerate(outFiles.items()):
                if k >= chunk.range.start and k <= chunk.range.last:
                    img, h_ori, w_ori, PAR, orientation = image.loadImage(iFile, True)

                    chunk.logger.info("image: {}".format(iFile))
                     
                    if not processFullImages:
                        bboxes = np.asarray(bboxDict[iFile]["bboxes"])
                        chunk.logger.debug("bboxes: {}".format(bboxDict[iFile]["bboxes"]))
                    else:
                        bboxes = np.array([[0, 0, img.shape[1] - 1, img.shape[0] - 1]])

                    mask = processor.process(image = img,
                                             bboxes = bboxes,
                                             invert = chunk.node.maskInvert.value,
                                             verbose = False)

                    image.writeImage(oFile[0], mask, h_ori, w_ori, orientation, PAR)

            del processor
            torch.cuda.empty_cache()

        finally:
            chunk.logManager.end()

    def stopProcess(sel, chunk):
        # torch.cuda.empty_cache()
        try:
            del processor
        except:
            pass

