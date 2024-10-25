__version__ = "0.1"

from meshroom.core import desc
import os


class ImageDetectionPrompt(desc.Node):
    size = desc.DynamicNodeSize('input')
    gpu = desc.Level.INTENSIVE
    parallelization = desc.Parallelization(blockSize=50)

    category = 'Utils'
    documentation = '''
Generate bounded boxes corresponding to the input text prompt.
First a recognition model (image to tags) is launched on the input image.
If the prompt or a synonym is detected in the returned list of tags the detection model (tag to bounded box) is launched.
Detection can be forced by setting to True the appropriate parameter.
Bounded box sizes can be increased by a ratio from 0 to 100%
'''

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="SfMData file input.",
            value="",
        ),
        desc.File(
            name="recognitionModelPath",
            label="Recognition Model",
            description="Weights file for the recognition model.",
            value=os.getenv('RDS_RECOGNITION_MODEL_PATH',""),
        ),
        desc.File(
            name="detectionModelPath",
            label="Detection Model",
            description="Weights file for the detection model.",
            value=os.getenv('RDS_DETECTION_MODEL_PATH',""),
        ),
        desc.File(
            name="detectionConfigPath",
            label="Detection Config",
            description="Config file for the detection model.",
            value=os.getenv('RDS_DETECTION_CONFIG_PATH',""),
        ),
        desc.StringParam(
            name="prompt",
            label="Prompt",
            description="What to segment, separated by point or one item per line",
            value="person",
            semantic="multiline",
        ),
        desc.StringParam(
            name="synonyms",
            label="Synonyms",
            description="Synonyms to prompt separated by commas or one item per line. eg: man,woman,boy,girl,human,people can be used as synonyms of person",
            value="man\nwoman\nboy\ngirl\nhuman\npeople",
            semantic="multiline",
        ),
        desc.BoolParam(
            name="forceDetection",
            label="Force Detection",
            description="Launch detection step even if nor the prompt neither any synonyms are recognized",
            value=False,
        ),
        desc.FloatParam(
            name="thresholdDetection",
            label="Threshold Detection",
            description="Threshold for detection, the lower the more sensitive the detector",
            range=(0.0,1.0,0.1),
            value=0.2,
        ),
        desc.IntParam(
            name="bboxMargin",
            label="Detection Margin",
            description="Increase bounded box dimensions by the selected percentage",
            range=(0,100,1),
            value=0,
        ),
        desc.BoolParam(
            name="useGpu",
            label="Use GPU",
            description="Use GPU for computation if available",
            value=True,
            invalidate=False,
        ),
        desc.BoolParam(
            name="outputBboxImage",
            label="Output Bounded Box Image",
            description="Write source image with bounded boxes baked in.",
            value=False,
        ),
        desc.BoolParam(
            name="keepFilename",
            label="Keep Filename",
            description="Keep Input Filename",
            value=False,
            enabled=lambda node: node.outputBboxImage.value,
        ),
        desc.ChoiceParam(
            name="extension",
            label="Output File Extension",
            description="Output image file extension.",
            value="jpg",
            values=["png", "jpg"],
            exclusive=True,
            enabled=lambda node: node.outputBboxImage.value,
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
            label="BBox Folder",
            description="Output path for the bounded boxes.",
            value=desc.Node.internalFolder,
        ),
        desc.File(
            name="bboxes",
            label="BBoxes",
            description="Generated images with bounded boxes baked in.",
            semantic="image",
            value=lambda attr: desc.Node.internalFolder + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + "." + attr.node.extension.value,
            group="",
        ),
    ]

    def resolvedPaths(self, inputSfm, outDir, keepFilename, ext):
        import pyalicevision as av
        from pathlib import Path

        paths = {}
        dataAV = av.sfmData.SfMData()
        if av.sfmDataIO.load(dataAV, inputSfm, av.sfmDataIO.ALL) and os.path.isdir(outDir):
            views = dataAV.getViews()
            for id, v in views.items():
                inputFile = v.getImage().getImagePath()
                if keepFilename:
                    outputFileMask = os.path.join(outDir, Path(inputFile).stem + '.' + ext)
                    outputFileBoxes = os.path.join(outDir, Path(inputFile).stem + "_bboxes" + '.jpg')
                else:
                    outputFileMask = os.path.join(outDir, str(id) + '.' + ext)
                    outputFileBoxes = os.path.join(outDir, str(id) + "_bboxes" + '.jpg')
                paths[inputFile] = (outputFileMask, outputFileBoxes)

        return paths

    def processChunk(self, chunk):
        import json
        from segmentationRDS import image, segmentation
        import torch

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)

            if not chunk.node.input:
                chunk.logger.warning('Nothing to segment')
                return
            if not chunk.node.output.value:
                return

            chunk.logger.info('Chunk range from {} to {}'.format(chunk.range.start, chunk.range.last))

            outFiles = self.resolvedPaths(chunk.node.input.value, chunk.node.output.value, chunk.node.keepFilename.value, chunk.node.extension.value)

            if not os.path.exists(chunk.node.output.value):
                os.mkdir(chunk.node.output.value)

            os.environ['TOKENIZERS_PARALLELISM'] = 'true' # required to avoid warning on tokenizers

            processor = segmentation.detectAnything(RAM_CHECKPOINT_PATH = chunk.node.recognitionModelPath.value,
                                                    GD_CONFIG_PATH = chunk.node.detectionConfigPath.value,
                                                    GD_CHECKPOINT_PATH = chunk.node.detectionModelPath.value,
                                                    useGPU = chunk.node.useGpu.value)

            prompt = chunk.node.prompt.value.replace('\n','.')
            chunk.logger.debug('prompt: {}'.format(prompt))
            synonyms = chunk.node.synonyms.value.replace('\n',',')
            chunk.logger.debug('synonyms: {}'.format(synonyms))

            dict = {}

            for k, (iFile, oFile) in enumerate(outFiles.items()):
                if k >= chunk.range.start and k <= chunk.range.last:
                    img, h_ori, w_ori, PAR = image.loadImage(iFile, True)
                    bboxes, conf, tags = processor.process(image = img,
                                                           prompt = chunk.node.prompt.value,
                                                           synonyms = chunk.node.synonyms.value,
                                                           threshold = chunk.node.thresholdDetection.value,
                                                           force = chunk.node.forceDetection.value,
                                                           bboxMargin = chunk.node.bboxMargin.value,
                                                           verbose = False)

                    chunk.logger.debug('image: {}'.format(iFile))
                    chunk.logger.debug('tags: {}'.format(tags))
                    chunk.logger.debug('bboxes: {}'.format(bboxes))
                    chunk.logger.debug('confidence: {}'.format(conf))

                    imgInfo = {}
                    imgInfo['bboxes'] = bboxes.tolist()
                    imgInfo['confidence'] = conf.tolist()
                    imgInfo['tags'] = tags
                    imgInfo['prompt'] = prompt.split('.')
                    imgInfo['synonyms'] = synonyms.split(',')
                    dict[iFile] = imgInfo

                    if (chunk.node.outputBboxImage.value):
                        imgBoxes = (img * 255.0).astype('uint8')
                        for bbox in bboxes:
                            imgBoxes = image.addRectangle(imgBoxes, bbox)
                        image.writeImage(oFile[0], imgBoxes, h_ori, w_ori, PAR)

            jsonFilename = chunk.node.output.value + "/bboxes." + str(chunk.index) + ".json"
            jsonObject = json.dumps(dict, indent = 4)
            outfile =  open(jsonFilename, 'w')
            outfile.write(jsonObject)
            outfile.close()

            del processor
            torch.cuda.empty_cache()

        finally:
            chunk.logManager.end()

    def stopProcess(sel, chunk):
        try:
            del processor
        except:
            pass

