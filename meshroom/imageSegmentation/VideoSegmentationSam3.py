__version__ = "0.3"

import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class Sam3VideoNodeSize(desc.MultiDynamicNodeSize):
    def computeSize(self, node):
        # if node.attribute(self._params[0]).isLink:
        #     return node.attribute(self._params[0]).inputLink.node.size

        # from pathlib import Path

        # input_path_param = node.attribute(self._params[0])
        # extension_param = node.attribute(self._params[1])
        # input_path = input_path_param.value
        # extension = extension_param.value
        # include_suffixes = [extension.lower(), extension.upper()]

        size = 1
        # if Path(input_path).is_dir():
        #     import itertools
        #     image_paths = list(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
        #     size = len(image_paths)
        
        return size
        
class VideoSegmentationSam3(desc.Node):
    size = Sam3VideoNodeSize(['input', 'extensionIn'])
    gpu = desc.Level.INTENSIVE
    #parallelization = desc.Parallelization(blockSize=50)

    category = "Utils"
    documentation = """
Based on the Segment Anything model 3, the node generates a binary mask from a text prompt.
It is strongly advised to launch a first segmentation using only a text prompt.
Two masks are generated, a binary one and a colored one that the indexes of every sub masks.
Object Ids are color encoded as follow:
 0:[1,0,0] = xff0000
 1:[0,1,0] = x00ff00
 2:[0,0,1] = x0000ff
 3:[1,1,0] = xffff00
 4:[1,0,1] = xff00ff
 5:[0,1,1] = x00ffff
 6:[1,0,0.5] = xff0080
 7:[0,1,0.5] = x00ff80
 8:[0,0.5,1] = x0080ff
 9:[1,1,0.5] = xffff80
10:[1,0.5,1] = xff80ff
11:[0.5,1,1] = x80ffff
12:[1,0.5,0] = xff8000
13:[0.5,1,0] = x80ff00
14:[0.5,0,1] = x8000ff
15:[1,0.5,0.5] = xff8080
16:[0.5,1,0.5] = x80ff80
17:[0.5,0.5,1] = x8080ff
18:[1,1,1] = xffffff
After that, refinement is possible through in/out points for every segmented objects.
In order to associate a point to a given sub mask, it must be colored with the corresponding color.
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
            description="What to segment, separated by point or one item per line.",
            value="person",
            semantic="multiline",
        ),
        desc.File(
            name="bboxFolder",
            label="BBoxes Folder",
            description="JSON file containing prompting bounding boxes.",
            value="",
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
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug).",
            value="info",
            values=VERBOSE_LEVEL,
            exclusive=True,
        ),
        desc.ShapeList(
            name="positiveClicks",
            label="Positive Clicks",
            description="Prompt: Positive Clicks",
            shape=desc.Point2d(
                name="click",
                label="Click",
                description="Point.",
                keyable=True,
                keyType="viewId",
            ),
        ),
        desc.ShapeList(
            name="negativeClicks",
            label="Negative Clicks",
            description="Prompt: Negative Clicks",
            shape=desc.Point2d(
                name="click",
                label="Click",
                description="Point.",
                keyable=True,
                keyType="viewId",
            ),
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
        desc.File(
            name="colorMasks",
            label="Colored Masks",
            description="Generated segmentation masks with color corresponding to item indexes.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/colorMask_" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + ".png",
            group="",
        ),
    ]

    def prepare_masks_for_visualization(self, frame_to_output):
        # frame_to_obj_masks --> {frame_idx: {'output_probs': np.array, `out_obj_ids`: np.array, `out_binary_masks`: np.array}}
        for frame_idx, out in frame_to_output.items():
            _processed_out = {}
            for idx, obj_id in enumerate(out["out_obj_ids"].tolist()):
                if out["out_binary_masks"][idx].any():
                    _processed_out[obj_id] = out["out_binary_masks"][idx]
            frame_to_output[frame_idx] = _processed_out
        return frame_to_output

    def propagate_in_video(self, predictor, session_id):
        # we will just propagate from frame 0 to the end of the video
        outputs_per_frame = {}
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
        return outputs_per_frame
        
    def getClickDictWithViewIdAsKeyFromShape(self, shape):
        clickDictFromShape = {}
        shapesClicksIn = shape.getShapesAsDict()
        if shapesClicksIn:
            for sh in shapesClicksIn:
                color = sh["properties"]["color"]
                for key in sh["observations"]:
                    x = sh["observations"][key]["x"]
                    y = sh["observations"][key]["y"]
                    pt = [x, y]
                    if key in clickDictFromShape:
                        clickDictFromShape[key].append((pt, color))
                    else:
                        clickDictFromShape[key] = [(pt, color)]
        return clickDictFromShape

    def normalize_click(self, click_xy, img_w, img_h, PAR, orientation):
        from segmentationRDS import image
        normalized_click = click_xy.copy()
        normalized_click[0], normalized_click[1] = image.fromRawToUsualOrientation(normalized_click[0], normalized_click[1], img_w, img_h, PAR, orientation)
        normalized_click[0] /= img_w
        normalized_click[1] /= img_h
        return normalized_click

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
                    bb = [xc - w/2, yc - h/2, w, h]
                    if key in bboxDictFromShape:
                        bboxDictFromShape[key].append(bb)
                    else:
                        bboxDictFromShape[key] = [bb]
        return bboxDictFromShape

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


    def preprocess(self, node):
        extension = node.extensionIn.value
        input_path = node.input.value
        image_paths = get_image_paths_list(input_path, extension)
        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {input_path}')
        self.image_paths = image_paths


    def processChunk(self, chunk):
        import json
        from segmentationRDS import image
        from sam3.model_builder import build_sam3_video_predictor
        import numpy as np
        import torch
        from pyalicevision import image as avimg
        from PIL import Image

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)

            if not chunk.node.input:
                chunk.logger.warning("Nothing to segment")
                return
            if not chunk.node.output.value:
                return

            chunk.logger.info("Chunk range from {} to {}".format(chunk.range.start, chunk.range.last))

            chunk_image_paths = self.image_paths

            if not os.path.exists(chunk.node.output.value):
                os.mkdir(chunk.node.output.value)

            gpus_to_use = [torch.cuda.current_device()]
            video_predictor = build_sam3_video_predictor(checkpoint_path=chunk.node.segmentationModelPath.evalValue, gpus_to_use=gpus_to_use)

            posClickDictFromShape = self.getClickDictWithViewIdAsKeyFromShape(chunk.node.positiveClicks)
            negClickDictFromShape = self.getClickDictWithViewIdAsKeyFromShape(chunk.node.negativeClicks)
            posBboxDictFromShape = self.getBboxDictWithViewIdAsKeyFromShape(chunk.node.positiveBoxes)
            negBboxDictFromShape = self.getBboxDictWithViewIdAsKeyFromShape(chunk.node.negativeBoxes)

            metadata_deep_model = {}
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelName"] = "SegmentAnything"
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelVersion"] = "sam3-Video"

            pil_images = []
            clicks = {}
            bboxes = {}

            colors=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],
                    [255,0,128],[0,255,128],[0,128,255],[255,255,128],[255,128,255],[128,255,255],
                    [255,128,0],[128,255,0],[128,0,255],[255,128,128],[128,255,128],[128,128,255],[255,255,255]]
            
            for idx, path in enumerate(chunk_image_paths):
                img, h_ori, w_ori, PAR, orientation = image.loadImage(str(chunk_image_paths[idx][0]), True)
                pil_images.append(Image.fromarray((255.0*img).astype("uint8")))
                sourceInfo = {"h_ori": h_ori, "w_ori": w_ori, "PAR": PAR, "orientation": orientation}

                viewId = chunk_image_paths[idx][1]
                frameId = chunk_image_paths[idx][2]

                objects = {}
                if viewId is not None and str(viewId) in posClickDictFromShape:
                    for pt in posClickDictFromShape[viewId]:
                        color = [int(pt[1][1:3], 16), int(pt[1][3:5], 16), int(pt[1][5:], 16)]
                        if color not in colors:
                            colors.append(color)
                        objId = colors.index(color)

                        if objId not in objects:
                            objects[objId] = [[], []]

                        p = self.normalize_click(pt[0], img.shape[1], img.shape[0], PAR, orientation)
                        objects[objId][0].append(p)
                        objects[objId][1].append(1)

                if viewId is not None and str(viewId) in negClickDictFromShape:
                    for pt in negClickDictFromShape[viewId]:
                        color = [int(pt[1][1:3], 16), int(pt[1][3:5], 16), int(pt[1][5:], 16)]
                        if color not in colors:
                            colors.append(color)
                        objId = colors.index(color)

                        if objId not in objects:
                            objects[objId] = [[], []]

                        p = self.normalize_click(pt[0], img.shape[1], img.shape[0], PAR, orientation)
                        objects[objId][0].append(p)
                        objects[objId][1].append(0)

                if len(objects) > 0:
                    clicks[frameId] = objects

                if viewId is not None and str(viewId) in posBboxDictFromShape:
                    if frameId not in bboxes:
                        bboxes[frameId] = ([],[])
                    for bbox in posBboxDictFromShape[viewId]:
                        bbox = self.normalize_bbox(bbox, img.shape[1], img.shape[0], PAR, orientation)
                        bboxes[frameId][0].append(bbox)
                        bboxes[frameId][1].append(1)

                if viewId is not None and str(viewId) in negBboxDictFromShape:
                    if frameId not in bboxes:
                        bboxes[frameId] = ([],[])
                    for bbox in negBboxDictFromShape[viewId]:
                        bbox = self.normalize_bbox(bbox, img.shape[1], img.shape[0], PAR, orientation)
                        bboxes[frameId][0].append(bbox)
                        bboxes[frameId][1].append(0)

            chunk.logger.debug(f"clicks = {clicks}")
            chunk.logger.debug(f"bboxes = {bboxes}")

            response = video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=pil_images,
                    )
            )
            session_id = response["session_id"]

            #if chunk.node.prompt.value != "":
            response = video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=chunk.node.prompt.value,
                )
            )

            for f, bbox in bboxes.items():
                response = video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=f,
                        bounding_boxes=bbox[0],
                        bounding_box_labels=bbox[1],
                    )
                )

            outputs_per_frame = self.propagate_in_video(video_predictor, session_id)

            for f, objects in clicks.items():
                for obj_id, obj in objects.items():
                    response = video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=f,
                            points=torch.tensor(np.array(obj[0])),
                            point_labels=torch.tensor(np.array(obj[1])),
                            obj_id = obj_id
                        )
                    )

            outputs_per_frame = self.propagate_in_video(video_predictor, session_id)

            outputs_per_frame = self.prepare_masks_for_visualization(outputs_per_frame)

            video_predictor.handle_request(request=dict(type="close_session", session_id=session_id))

            for frameId, masks in outputs_per_frame.items():
                maskImage = np.zeros_like(img)
                colorMaskImage = np.zeros_like(img)
                for key, mask in masks.items():
                    maskImage[mask] = [255, 255, 255]
                    colorMaskImage[mask] = [x/255.0 for x in colors[int(key) % len(colors)]]

                if chunk.node.maskInvert.value:
                    mask = (maskImage[:,:,0:1] == 0).astype('float32')
                else:
                    mask = (maskImage[:,:,0:1] > 0).astype('float32')
                chunk.logger.info("frameId: {} - {}".format(frameId, chunk_image_paths[frameId][0]))

                if chunk.node.keepFilename.value:
                    outputFileMask = os.path.join(chunk.node.output.value, Path(chunk_image_paths[frameId][0]).stem + "." + chunk.node.extensionOut.value)
                    outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + str(Path(chunk_image_paths[frameId][0]).stem) + ".png")
                else:
                    outputFileMask = os.path.join(chunk.node.output.value, str(chunk_image_paths[frameId][1]) + "." + chunk.node.extensionOut.value)
                    outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + str(chunk_image_paths[frameId][1]) + ".png")

                optWrite = avimg.ImageWriteOptions()
                optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
                if Path(outputFileMask).suffix.lower() == ".exr":
                    optWrite.exrCompressionMethod(avimg.EImageExrCompression_stringToEnum("DWAA"))
                    optWrite.exrCompressionLevel(300)

                image.writeImage(outputFileMask, mask, sourceInfo["h_ori"], sourceInfo["w_ori"], sourceInfo["orientation"], sourceInfo["PAR"], metadata_deep_model, optWrite)
                image.writeImage(outputFileColorMask, colorMaskImage, sourceInfo["h_ori"], sourceInfo["w_ori"], sourceInfo["orientation"], sourceInfo["PAR"], metadata_deep_model, optWrite)

        finally:
            torch.cuda.empty_cache()
            chunk.logManager.end()


def get_image_paths_list(input_path, extension):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO
    from pathlib import Path
    import itertools

    include_suffixes = [extension.lower(), extension.upper()]
    image_paths = []

    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
        image_paths = [(p, None, None) for p in image_paths]
    elif Path(input_path).suffix.lower() in [".sfm", ".abc"]:
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                for id, v in views.items():
                   image_paths.append((Path(v.getImage().getImagePath()), str(id), v.getFrameId()))

            image_paths.sort(key=lambda x: x[0])
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path (folder or sfmData file).")
    return image_paths