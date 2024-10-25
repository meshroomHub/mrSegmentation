__version__ = "0.1"

from meshroom.core import desc
import os


class ImageSegmentationBox(desc.Node):
    size = desc.DynamicNodeSize('input')
    gpu = desc.Level.INTENSIVE
    parallelization = desc.Parallelization(blockSize=50)

    category = 'Utils'
    documentation = '''
Generate a binary mask from an input bounded box and points.
Based on the Segment Anything model.
'''

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
            name="segmentationModelPath",
            label="Segmentation Model",
            description="Weights file for the segmentation model.",
            value=os.getenv('RDS_SEGMENTATION_MODEL_PATH'),
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
            description="Use GPU for computation if available",
            value=True,
            invalidate=False,
        ),
        desc.BoolParam(
            name="keepFilename",
            label="Keep Filename",
            description="Keep Input Filename",
            value=False,
        ),
        desc.ChoiceParam(
            name="extension",
            label="Output File Extension",
            description="Output image file extension.\n"
                        "If unset, the output file extension will match the input's if possible.",
            value="exr",
            values=["exr", "png", "jpg"],
            exclusive=True,
            group='', # remove from command line params
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug).",
            value="info",
            values=["fatal", "error", "warning", "info", "debug"],
            exclusive=True,
        )
    ]

    outputs = [
        desc.File(
            name="output",
            label="Masks Folder",
            description="Output path for the masks.",
            value=desc.Node.internalFolder,
        ),
        desc.File(
            name="masks",
            label="Masks",
            description="Generated segmentation masks.",
            semantic="image",
            value=lambda attr: desc.Node.internalFolder + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + "." + attr.node.extension.value,
            group="",
        ),
    ]

    def resolvedPaths(self, inputSfm, outDir, keepFilename):
        import pyalicevision as av
        from pathlib import Path

        paths = {}
        dataAV = av.sfmData.SfMData()
        if av.sfmDataIO.load(dataAV, inputSfm, av.sfmDataIO.ALL) and os.path.isdir(outDir):
            views = dataAV.getViews()
            for id, v in views.items():
                inputFile = v.getImage().getImagePath()
                if keepFilename:
                    outputFileMask = os.path.join(outDir, Path(inputFile).stem + '.exr')
                    outputFileBoxes = os.path.join(outDir, "bboxes_" + Path(inputFile).stem + '.jpg')
                else:
                    outputFileMask = os.path.join(outDir, str(id) + '.exr')
                    outputFileBoxes = os.path.join(outDir, "bboxes_" + str(id) + '.exr')
                paths[inputFile] = (outputFileMask, outputFileBoxes)

        return paths

    def processChunk(self, chunk):
        import json
        from segmentationRDS import image, segmentation
        import numpy as np
        import torch

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)

            if not chunk.node.input:
                chunk.logger.warning('Nothing to segment')
                return
            if not chunk.node.bboxFolder.value:
                chunk.logger.warning('No folder containing bounded boxes')
                return
            if not chunk.node.output.value:
                return

            chunk.logger.info('Chunk range from {} to {}'.format(chunk.range.start, chunk.range.last))

            outFiles = self.resolvedPaths(chunk.node.input.value, chunk.node.output.value, chunk.node.keepFilename.value)

            if not os.path.exists(chunk.node.output.value):
                os.mkdir(chunk.node.output.value)

            processor = segmentation.SegmentAnything(SAM_CHECKPOINT_PATH = chunk.node.segmentationModelPath.value,
                                                     useGPU = chunk.node.useGpu.value)

            bboxDict = {}
            for file in os.listdir(chunk.node.bboxFolder.value):
                if file.endswith('.json'):
                    with open(os.path.join(chunk.node.bboxFolder.value,file)) as bboxFile:
                        bb = json.load(bboxFile)
                        bboxDict.update(bb)

            for k, (iFile, oFile) in enumerate(outFiles.items()):
                if k >= chunk.range.start and k <= chunk.range.last:
                    img, h_ori, w_ori, PAR = image.loadImage(iFile, True)
                    bboxes = np.asarray(bboxDict[iFile]['bboxes'])

                    chunk.logger.info('image: {}'.format(iFile))
                    chunk.logger.debug('bboxes: {}'.format(bboxDict[iFile]['bboxes']))
                    
                    mask = processor.process(image = img,
                                             bboxes = bboxes,
                                             clicksIn = [],
                                             clicksOut = [],
                                             invert = chunk.node.maskInvert.value,
                                             verbose = False)

                    image.writeImage(oFile[0], mask, h_ori, w_ori, PAR)

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

