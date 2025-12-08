__version__ = "0.1"

import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class Sam3NodeSize(desc.MultiDynamicNodeSize):
    def computeSize(self, node):
        if node.attribute(self._params[0]).isLink:
            return node.attribute(self._params[0]).inputLink.node.size

        from pathlib import Path

        input_path_param = node.attribute(self._params[0])
        extension_param = node.attribute(self._params[1])
        input_path = input_path_param.value
        extension = extension_param.value
        include_suffixes = [extension.lower(), extension.upper()]

        size = 1
        if Path(input_path).is_dir():
            import itertools
            image_paths = list(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
            size = len(image_paths)
        
        return size
        
class Sam3(desc.Node):
    size = Sam3NodeSize(['input', 'extensionIn'])
    gpu = desc.Level.INTENSIVE
    parallelization = desc.Parallelization(blockSize=50)

    category = "Utils"
    documentation = """
Based on the Segment Anything model 3, the node generates a binary mask from a text prompt and a set of bounding boxes.
The bounding boxes can be provided through a json file and loaded by clicking on a push button or manualy defined on the 2D viewer.
When loaded from a json file containing rectangle shapes, the lowered shape name must contains the substring "pos" for the positive bounding boxes and "neg" for the negative ones.
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
            enabled=lambda node: Path(node.input.value).is_dir(),
        ),
        desc.StringParam(
            name="prompt",
            label="Prompt",
            description="Text prompt, 1 concept per line.",
            value="person\nchild\npeople",
            semantic="multiline",
        ),
        desc.File(
            name="bboxFolder",
            label="BBoxes Folder",
            description="JSON file containing prompting bounding boxes.",
            value="",
            invalidate=False,
        ),
        desc.PushButtonParam(
            name="bboxLoad",
            label="Load Boxes",
            description="Load input shape files containing bounding boxes",
        ),
        desc.File(
            name="segmentationModelPath",
            label="Segmentation Model",
            description="Weights file for the segmentation model.",
            value="${RDS_SAM3_MODEL_PATH}",
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
            enabled=lambda node: not Path(node.input.value).is_dir(),
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
            name="splitBoxPrompt",
            label="Split Box Prompt",
            description="Reset detector before feeding with a new positive box and merge results. Negative boxes will be ignored.",
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
        desc.ShapeList(
            name="positiveBoxes",
            label="Positive Boxes",
            description="Prompt: Positive Bounding Boxes",
            shape=desc.Rectangle(
                name="bbox",
                label="Bounding Box",
                description="Rectangle.",
                keyable=True,
                keyType="viewId",
            ),
        ),
        desc.ShapeList(
            name="negativeBoxes",
            label="Negative Boxes",
            description="Prompt: Negative Bounding Boxes",
            shape=desc.Rectangle(
                name="bbox",
                label="Bounding Box",
                description="Rectangle.",
                keyable=True,
                keyType="viewId",
            ),
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
    ]

    def onBboxLoadClicked(self, node):
        import json
        from pathlib import Path
        if node.bboxFolder.value:
            shapeFiles = list(Path(node.bboxFolder.value).glob("*shapes.json"))
            if len(shapeFiles) > 0:
                # remove all existing input shapes
                node.positiveBoxes.resetToDefaultValue()
                node.negativeBoxes.resetToDefaultValue()
            for file in shapeFiles:
                with open(os.path.join(node.bboxFolder.value,file)) as shapeFile:
                    listListshapes = json.load(shapeFile)
                    for k,shapes in enumerate(listListshapes):
                        if len(shapes) > 0:
                            for shape in shapes:
                                if shape["type"] == "Rectangle" and ("pos" in shape["name"].lower() or "neg" in shape["name"].lower()):
                                    attrDict = {"userName": shape["name"], "userColor": shape["properties"]["color"], "geometry": {}}
                                    attrDict["geometry"]["center"]={"x":{}, "y":{}}
                                    attrDict["geometry"]["size"]={"width":{}, "height":{}}
                                    for key in shape["observations"]:
                                        attrDict["geometry"]["center"]["x"][key] = shape["observations"][key]["center"]["x"]
                                        attrDict["geometry"]["center"]["y"][key] = shape["observations"][key]["center"]["y"]
                                        attrDict["geometry"]["size"]["width"][key] = shape["observations"][key]["size"]["width"]
                                        attrDict["geometry"]["size"]["height"][key] = shape["observations"][key]["size"]["height"]
                                    if "pos" in shape["name"].lower():
                                        node.positiveBoxes.insert(k, attrDict)
                                    elif "neg" in shape["name"].lower():
                                        node.negativeBoxes.insert(k, attrDict)

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
                paths[str(inputFile)] = (outputFileMask, outputFileBoxes, frameId, 'not_a_view')
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
                        paths[inputFile] = (outputFileMask, outputFileBoxes, frameId, str(id))

        return paths

    def normalize_bbox(self, bbox_xywh, img_w, img_h, PAR, orientation):
        import torch
        from segmentationRDS import image
        # Assumes bbox_xywh is in XYWH format
        if isinstance(bbox_xywh, list):
            assert (
                len(bbox_xywh) == 4
            ), "bbox_xywh list must have 4 elements. Batching not support except for torch tensors."
            normalized_bbox = bbox_xywh.copy()
            normalized_bbox[0], normalized_bbox[1] = image.fromRawToUsualOrientation(normalized_bbox[0], normalized_bbox[1], img_w, img_h, PAR, orientation)
            normalized_bbox[2], normalized_bbox[3] = image.fromRawToUsualOrientation(normalized_bbox[2], normalized_bbox[3], img_w, img_h, PAR, orientation)
            normalized_bbox[0] /= img_w
            normalized_bbox[1] /= img_h
            normalized_bbox[2] /= img_w
            normalized_bbox[3] /= img_h
        else:
            assert isinstance(
                bbox_xywh, torch.Tensor
            ), "Only torch tensors are supported for batching."
            normalized_bbox = bbox_xywh.clone()
            assert (
                normalized_bbox.size(-1) == 4
            ), "bbox_xywh tensor must have last dimension of size 4."
            normalized_bbox[..., 0], normalized_bbox[..., 1] = image.fromRawToUsualOrientation(normalized_bbox[..., 0], normalized_bbox[..., 1], img_w, img_h, PAR, orientation)
            normalized_bbox[..., 2], normalized_bbox[..., 3] = image.fromRawToUsualOrientation(normalized_bbox[..., 2], normalized_bbox[..., 3], img_w, img_h, PAR, orientation)
            normalized_bbox[..., 0] /= img_w
            normalized_bbox[..., 1] /= img_h
            normalized_bbox[..., 2] /= img_w
            normalized_bbox[..., 3] /= img_h
        return normalized_bbox

    def updateDetectedBboxes(self, detectedBBoxes, xyxy, idx, key, color):
        if idx+1 > len(detectedBBoxes):
            shape_bbox = {
                "name": "BBox_" + str(idx),
                "type": "Rectangle",
                "properties": {
                    "color": color},
                "observations" : {}}
            detectedBBoxes.append(shape_bbox)
        detectedBBoxes[idx]["observations"][key] = {
            "center" : {
                "x": (xyxy[0] + xyxy[2]) / 2,
                "y": (xyxy[1] + xyxy[3]) / 2
                },
            "size" : {
                "width": xyxy[2] - xyxy[0],
                "height": xyxy[3] - xyxy[1]
                }}

    def updateMaskImageAndDetectedBboxes(self, inference_state, maskImage, detectedBBoxes, key, w_ori, h_ori, PAR, orientation):
        from segmentationRDS import image
        masks, boxes, scores = inference_state["masks"], inference_state["boxes"], inference_state["scores"]
        for mask in masks:
            maskImage[mask.squeeze(0).cpu()] = [255, 255, 255]
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.cpu().tolist()
            x1, y1 = image.fromUsualToRawOrientation(x1, y1, w_ori, h_ori, PAR, orientation)
            x2, y2 = image.fromUsualToRawOrientation(x2, y2, w_ori, h_ori, PAR, orientation)
            self.updateDetectedBboxes(detectedBBoxes, [x1, y1, x2, y2], idx, key, "red")

    def getBboxDictWithViewIdAsKeyFromShape(self, shape):
        bboxDictFromShape = {}
        shapesBBoxesIn = shape.getShapesAsDict()
        if shapesBBoxesIn:
            for sh in shapesBBoxesIn:
                for key in sh["observations"]:
                    xc = sh["observations"][key]["center"]["x"]
                    yc = sh["observations"][key]["center"]["y"]
                    w = sh["observations"][key]["size"]["width"]
                    h = sh["observations"][key]["size"]["height"]
                    bb = [xc, yc, w, h]
                    if key in bboxDictFromShape:
                        bboxDictFromShape[key].append(bb)
                    else:
                        bboxDictFromShape[key] = [bb]
        return bboxDictFromShape

    def processChunk(self, chunk):
        import json
        import re
        from segmentationRDS import image
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import numpy as np
        import torch
        from pyalicevision import image as avimg

        processor = None
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)

            if not chunk.node.input:
                chunk.logger.warning("Nothing to segment")
                return
            if not chunk.node.output.value:
                return
            if not chunk.node.bboxFolder.value and not chunk.node.prompt.value:
                chunk.logger.warning("No Prompt, no output bounding boxes")

            chunk.logger.info("Chunk range from {} to {}".format(chunk.range.start, chunk.range.last))

            outFiles = self.resolvedPaths(chunk.node.input.value, chunk.node.extensionIn.value, chunk.node.output.value, chunk.node.keepFilename.value, chunk.node.extensionOut.value)

            if not os.path.exists(chunk.node.output.value):
                os.mkdir(chunk.node.output.value)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = build_sam3_image_model(checkpoint_path=chunk.node.segmentationModelPath.evalValue, device=device)
            processor = Sam3Processor(model)

            posBboxDictFromShape = self.getBboxDictWithViewIdAsKeyFromShape(chunk.node.positiveBoxes)
            negBboxDictFromShape = self.getBboxDictWithViewIdAsKeyFromShape(chunk.node.negativeBoxes)

            metadata_deep_model = {}
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelName"] = "SegmentAnything"
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelVersion"] = "sam3"

            textPrompts = re.split(r'[\n]+', chunk.node.prompt.value)
            textPrompts = [textPrompt for textPrompt in textPrompts if textPrompt]

            detectedShapeBboxes = []

            for k, (iFile, oFile) in enumerate(outFiles.items()):
                if k >= chunk.range.start and k <= chunk.range.last:
                    img, h_ori, w_ori, PAR, orientation = image.loadImage(iFile, True)
                    frameId = oFile[2]
                    viewId = oFile[3]
                    key = iFile if viewId == "not_a_view" else viewId

                    chunk.logger.info("frameId: {} - {}".format(frameId, iFile))

                    bboxes = []
                    bboxLabels = []
                    if viewId != "not_a_view" and viewId in posBboxDictFromShape:
                        for bbox in posBboxDictFromShape[viewId]:
                            bbox = self.normalize_bbox(bbox, img.shape[1], img.shape[0], PAR, orientation)
                            bboxes.append(bbox)
                            bboxLabels.append(True)
                    if viewId != "not_a_view" and viewId in negBboxDictFromShape:
                        for bbox in negBboxDictFromShape[viewId]:
                            bbox = self.normalize_bbox(bbox, img.shape[1], img.shape[0], PAR, orientation)
                            bboxes.append(bbox)
                            bboxLabels.append(False)

                    chunk.logger.debug(f"text prompts: {textPrompts}")
                    chunk.logger.debug(f"bboxes: {bboxes}")
                    chunk.logger.debug(f"bboxLabels: {bboxLabels}")

                    imgTensor = torch.from_numpy(np.transpose((255.0*img).astype(np.uint8), (2, 0, 1)))
                    inference_state = processor.set_image(imgTensor)

                    mask_image = np.zeros_like(img)
                    # Reset processor
                    for textPrompt in textPrompts:
                        # Prompt the model with text
                        processor.reset_all_prompts(state=inference_state)
                        inference_state = processor.set_text_prompt(state=inference_state, prompt=textPrompt)
                        # Get the masks, bounding boxes, and scores
                        if "masks" in inference_state:
                            self.updateMaskImageAndDetectedBboxes(inference_state, mask_image, detectedShapeBboxes, key, w_ori, h_ori, PAR, orientation)

                    if not chunk.node.splitBoxPrompt:
                        processor.reset_all_prompts(state=inference_state)
                    for box, label in zip(bboxes, bboxLabels):
                        # Prompt the model with bboxes
                        if chunk.node.splitBoxPrompt:
                            processor.reset_all_prompts(state=inference_state)
                        inference_state = processor.add_geometric_prompt(state=inference_state, box=box, label=label)
                        # Get the masks, bounding boxes, and scores
                        if "masks" in inference_state and label and chunk.node.splitBoxPrompt:
                            self.updateMaskImageAndDetectedBboxes(inference_state, mask_image, detectedShapeBboxes, key, w_ori, h_ori, PAR, orientation)

                    if "masks" in inference_state and not chunk.node.splitBoxPrompt:
                        self.updateMaskImageAndDetectedBboxes(inference_state, mask_image, detectedShapeBboxes, key, w_ori, h_ori, PAR, orientation)

                    if chunk.node.maskInvert.value:
                        mask = (mask_image[:,:,0:1] == 0).astype('float32')
                    else:
                        mask = (mask_image[:,:,0:1] > 0).astype('float32')

                    optWrite = avimg.ImageWriteOptions()
                    if Path(oFile[0]).suffix.lower() == ".exr":
                        optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
                        optWrite.exrCompressionMethod(avimg.EImageExrCompression_stringToEnum("DWAA"))
                        optWrite.exrCompressionLevel(300)
                    else:
                        optWrite.toColorSpace(avimg.EImageColorSpace_SRGB)

                    image.writeImage(oFile[0], mask, h_ori, w_ori, orientation, PAR, metadata_deep_model, optWrite)

            shapeFilename = chunk.node.output.value + "/" + str(chunk.index) + ".shapes.json"
            with open(shapeFilename, 'w') as of_json:
                json.dump(detectedShapeBboxes, of_json, indent=4)

        finally:
            del processor
            torch.cuda.empty_cache()
            chunk.logManager.end()
