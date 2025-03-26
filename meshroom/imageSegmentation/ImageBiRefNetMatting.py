__version__ = "0.2"

import os

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class ImageBiRefNetMatting(desc.Node):
    size = desc.DynamicNodeSize("input")
    gpu = desc.Level.INTENSIVE
    parallelization = desc.Parallelization(blockSize=50)

    category = "Utils"
    documentation = """
Based on the BiRefNet model, the node generates a matte from a set of bounding boxes.

The bounding boxes can be provided through a json file as the one generated by the imageDetectionPrompt node or through a nk file storing a Tracker4 Nuke tracker.
The only requirement for the tracker file is about the tracks names. To be considered, a track name must contain at least one of the following patterns:
 * "bboxT_" or "bboxS".
 * "bboxT_": The tracking bounding box will be used.
 * "bboxS_": The searching bounding box will be used.

The bounding boxes coming from the Nuke tracker are appended to the ones coming from the json file if any.
In case neither tracker nor json file is available, the model is applied on the full image.
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
        desc.File(
            name="nukeTracker",
            label="Nuke Tracker",
            description="Nuke file .nk containing a tracker node.",
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
        desc.BoolParam(
            name="outputBboxImage",
            label="Output Bounding Box Image",
            description="Write source image with bounding boxes baked in.",
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
            value=lambda attr: "{nodeCacheFolder}/" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + "." + attr.node.extension.value,
            group="",
        ),
        desc.File(
            name="bboxes",
            label="BBoxes",
            description="Images with retained bounded boxes and clicks In/Out baked in.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/bboxes_" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + ".jpg",
            enabled=lambda node: node.outputBboxImage.value,
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
                frameId = v.getFrameId()
                if keepFilename:
                    outputFileMask = os.path.join(outDir, Path(inputFile).stem + "." + extension)
                    outputFileBoxes = os.path.join(outDir, "bboxes_" + Path(inputFile).stem + ".jpg")
                else:
                    outputFileMask = os.path.join(outDir, str(id) + "." + extension)
                    outputFileBoxes = os.path.join(outDir, "bboxes_" + str(id) + ".jpg")
                paths[inputFile] = (outputFileMask, outputFileBoxes, frameId)

        return paths

    def processChunk(self, chunk):
        import json
        from segmentationRDS import image, segmentation, nktracker
        import numpy as np
        import torch

        processor = None
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)

            if not chunk.node.input:
                chunk.logger.warning("Nothing to segment")
                return
            if not chunk.node.output.value:
                return
            if not chunk.node.nukeTracker.value and not chunk.node.bboxFolder.value:
                chunk.logger.warning("No bounding boxes and no tracker info, the full image will be segmented")
            if chunk.node.nukeTracker.value:
                tracker = nktracker.nkTracker(chunk.node.nukeTracker.value)
            else:
                tracker = None

            chunk.logger.info("Chunk range from {} to {}".format(chunk.range.start, chunk.range.last))

            outFiles = self.resolvedPaths(chunk.node.input.value, chunk.node.output.value, chunk.node.keepFilename.value, chunk.node.extension.value)

            if not os.path.exists(chunk.node.output.value):
                os.mkdir(chunk.node.output.value)

            processor = segmentation.BiRefNetSeg(modelType = chunk.node.birefnetModelType.value,
                                                 useGPU = chunk.node.useGpu.value)
            bboxDict = {}
            if chunk.node.bboxFolder.value:
                for file in os.listdir(chunk.node.bboxFolder.value):
                    if file.endswith(".json"):
                        with open(os.path.join(chunk.node.bboxFolder.value,file)) as bboxFile:
                            bb = json.load(bboxFile)
                            bboxDict.update(bb)

            for k, (iFile, oFile) in enumerate(outFiles.items()):
                if k >= chunk.range.start and k <= chunk.range.last:
                    img, h_ori, w_ori, PAR, orientation = image.loadImage(iFile, True)
                    frameId = oFile[2]

                    chunk.logger.info("frameId: {} - {}".format(frameId, iFile))
                     
                    bboxes = []
                    if not chunk.node.nukeTracker.value and not chunk.node.bboxFolder.value:
                        bboxes = [[0, 0, img.shape[1] - 1, img.shape[0] - 1]]
                    elif chunk.node.bboxFolder.value:
                        bboxes = bboxDict[iFile]["bboxes"]
                        
                    if tracker is not None:
                        data = tracker.getDataAtFrame(frameId, img.shape[0], PAR)
                        trackNames = tracker.getTrackNames()
                        for trackName in trackNames:
                            if data[trackName][0][0] is not None:
                                if trackName.find("bboxT_") != -1:
                                    bboxes.append(data[trackName][1])
                                if trackName.find("bboxS_") != -1:
                                    bboxes.append(data[trackName][2])

                    chunk.logger.debug("bboxes: {}".format(bboxes))

                    mask = processor.process(image = img,
                                             bboxes = np.asarray(bboxes),
                                             invert = chunk.node.maskInvert.value,
                                             verbose = False)

                    image.writeImage(oFile[0], mask, h_ori, w_ori, orientation, PAR)

                    if chunk.node.outputBboxImage.value:
                        bbox_img = img.copy()
                        for bbox in bboxes:
                            bbox_img = image.addRectangle(bbox_img, bbox, (0, 255, 0))
                        image.writeImage(oFile[1], bbox_img, h_ori, w_ori, orientation, PAR)

        finally:
            del processor
            torch.cuda.empty_cache()
            chunk.logManager.end()
