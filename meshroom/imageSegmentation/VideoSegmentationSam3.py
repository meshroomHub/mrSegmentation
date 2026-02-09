__version__ = "0.1"

import os
from pathlib import Path
import struct

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

import logging
logger = logging.getLogger("VideoSegmentationSam3")

class Sam3VideoNodeSize(desc.MultiDynamicNodeSize):
    def computeSize(self, node):
        size = 1
        return size
        
class VideoSegmentationSam3(desc.Node):
    size = Sam3VideoNodeSize(['input', 'extensionIn'])
    gpu = desc.Level.INTENSIVE

    category = "Utils"
    documentation = """
Based on the Segment Anything video predictor model 3, the node generates a binary mask, a colored mask and an exr cryptomatte
from a text prompt, a single bounding box or a set of positive and negative clicks (Clicks In/Out).
Text prompt and Clicks can be combined to refine results. For refinement, points must be associated to an existing submask.
In order to associate a point to a given submask, it must be colored with the submask's color.
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
            name="outputCryptomatte",
            label="Output Cryptomatte",
            description="Generate exr images containing cryptomatte to encode the segmentation results.",
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
        desc.Rectangle(
            name="boxPrompt",
            label="Box Prompt",
            description="Single bounding box used as initial prompt.",
            keyable=True,
            keyType="viewId"
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
        desc.File(
            name="cryptomatte",
            label="Cryptomatte",
            description="Cryptomatte embedded in exr images.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/cryptomatte_" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + ".exr",
            enabled=lambda node: node.outputCryptomatte.value,
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
        
    def getClickDictWithViewIdAsKeyFromShape(self, shapeList):
        clickDictFromShape = {}
        shapes = shapeList.getShapesAsDict()
        if shapes:
            for sh in shapes:
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
        sh = shape.getShapeAsDict()
        if sh:
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
            ), "bbox_xywh list must have 4 elements. Batching not supported except for torch tensors."
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

    def hash_name(self, name):
        import mmh3
        import numpy as np
        hash_32 = mmh3.hash(name, seed=0) & 0xFFFFFFFF
        f32_val = np.frombuffer(struct.pack('<I', hash_32), dtype=np.float32)[0]
        f32_hex = hex(struct.unpack('<I', struct.pack('<f', f32_val))[0])[2:]
        return f32_val, f32_hex, hash_32


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
        import OpenImageIO as oiio

        try:
            logger.setLevel(chunk.node.verboseLevel.value.upper())

            if not chunk.node.input:
                logger.warning("Nothing to segment")
                return
            if not chunk.node.output.value:
                return

            logger.info("Chunk range from {} to {}".format(chunk.range.start, chunk.range.last))

            chunk_image_paths = self.image_paths

            if not os.path.exists(chunk.node.output.value):
                os.mkdir(chunk.node.output.value)

            gpus_to_use = [torch.cuda.current_device()]
            video_predictor = build_sam3_video_predictor(checkpoint_path=chunk.node.segmentationModelPath.evalValue, gpus_to_use=gpus_to_use)

            posClickDictFromShape = self.getClickDictWithViewIdAsKeyFromShape(chunk.node.positiveClicks)
            negClickDictFromShape = self.getClickDictWithViewIdAsKeyFromShape(chunk.node.negativeClicks)
            posBboxDictFromShape = self.getBboxDictWithViewIdAsKeyFromShape(chunk.node.boxPrompt)

            metadata_deep_model = {}
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelName"] = "SegmentAnything"
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelVersion"] = "sam3-Video"

            pil_images = []
            clicks = {}
            bboxes = {}

            colorPalette = image.paletteGenerator()
            
            for idx, path in enumerate(chunk_image_paths):
                img, h_ori, w_ori, PAR, orientation = image.loadImage(str(chunk_image_paths[idx][0]), True)
                pil_images.append(Image.fromarray((255.0*img).astype("uint8")))
                sourceInfo = {"h_ori": h_ori, "w_ori": w_ori, "PAR": PAR, "orientation": orientation}

                viewId = chunk_image_paths[idx][1]
                frameId = chunk_image_paths[idx][2]

                objects = {}
                if viewId is not None and viewId in posClickDictFromShape:
                    for pt in posClickDictFromShape[viewId]:
                        color = (int(pt[1][1:3], 16), int(pt[1][3:5], 16), int(pt[1][5:], 16))
                        if colorPalette.index(color) is None:
                            colorPalette.add_color(color)
                        objId = colorPalette.index(color)

                        if objId not in objects:
                            objects[objId] = [[], []]

                        p = self.normalize_click(pt[0], img.shape[1], img.shape[0], PAR, orientation)
                        objects[objId][0].append(p)
                        objects[objId][1].append(1)

                if viewId is not None and viewId in negClickDictFromShape:
                    for pt in negClickDictFromShape[viewId]:
                        color = (int(pt[1][1:3], 16), int(pt[1][3:5], 16), int(pt[1][5:], 16))
                        if colorPalette.index(color) is None:
                            colorPalette.add_color(color)
                        objId = colorPalette.index(color)

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

            logger.debug(f"clicks = {clicks}")
            logger.debug(f"bboxes = {bboxes}")

            response = video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=pil_images,
                    )
            )
            session_id = response["session_id"]

            video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=chunk.node.prompt.value,
                )
            )

            for f, bbox in bboxes.items():
                video_predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=f,
                        bounding_boxes=bbox[0],
                        bounding_box_labels=bbox[1],
                    )
                )

            self.propagate_in_video(video_predictor, session_id)

            for f, objects in clicks.items():
                for obj_id, obj in objects.items():
                    video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=f,
                            points=torch.tensor(np.array(obj[0])),
                            point_labels=torch.tensor(np.array(obj[1])),
                            obj_id=obj_id
                        )
                    )

            outputs_per_frame = self.propagate_in_video(video_predictor, session_id)

            outputs_per_frame = self.prepare_masks_for_visualization(outputs_per_frame)

            video_predictor.handle_request(request=dict(type="close_session", session_id=session_id))

            for frameId, masks in outputs_per_frame.items():
                maskImage = np.zeros_like(img)
                colorMaskImage = np.zeros_like(img)
                if chunk.node.outputCryptomatte.value:
                    crypto_id = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                    crypto_cov = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                    crypto_zeros = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                    manifest = {}

                if len(masks.keys()) > 0:
                    colorPalette.generate_palette(max(masks.keys()) + 1)
                cryptoName = "object" if chunk.node.prompt.value == "" else chunk.node.prompt.value
                for key, mask in masks.items():
                    maskImage[mask] = [255, 255, 255]
                    color = colorPalette.at(int(key)) if colorPalette.at(int(key)) is not None else [255, 255, 255]
                    colorMaskImage[mask] = [x/255.0 for x in color]
                    if chunk.node.outputCryptomatte.value:
                        obj_name = f"{cryptoName}_{int(key)}"
                        f32_hash, hex_val, _ = self.hash_name(obj_name)
                        manifest[obj_name] = hex_val
                        crypto_id[mask] = f32_hash
                        crypto_cov[mask] = 1.0

                if chunk.node.outputCryptomatte.value:
                    spec = oiio.ImageSpec(img.shape[1], img.shape[0], 7, oiio.FLOAT)
                    spec.channelnames = (cryptoName+".red", cryptoName+".green", cryptoName+".blue",
                                        cryptoName+"00.red", cryptoName+"00.green", cryptoName+"00.blue", cryptoName+"00.alpha")
                    _, _, h32 = self.hash_name(cryptoName)
                    crypto_key = f"{h32 & 0xFFFFFFFF:08x}"[:7]
                    spec.attribute(f"cryptomatte/{crypto_key}/name", cryptoName)
                    spec.attribute(f"cryptomatte/{crypto_key}/manifest", json.dumps(manifest))
                    spec.attribute(f"cryptomatte/{crypto_key}/hash", "MurmurHash3_32")
                    spec.attribute(f"cryptomatte/{crypto_key}/conversion", "uint32_to_float32")

                    if chunk.node.keepFilename.value:
                        cryptomattePath = os.path.join(chunk.node.output.value, "cryptomatte_" + str(Path(chunk_image_paths[frameId][0]).stem) + ".exr")
                    else:
                        cryptomattePath = os.path.join(chunk.node.output.value, "cryptomatte_" + str(chunk_image_paths[frameId][1]) + ".exr")

                    cryptomatteImg = oiio.ImageOutput.create(str(cryptomattePath))
                    cryptomatteImg.open(cryptomattePath, spec)
                    cryptomatte_data = np.dstack((crypto_zeros, crypto_zeros, crypto_zeros, crypto_id, crypto_cov, crypto_zeros, crypto_zeros))
                    cryptomatteImg.write_image(cryptomatte_data)
                    cryptomatteImg.close()

                if chunk.node.maskInvert.value:
                    mask = (maskImage[:,:,0:1] == 0).astype('float32')
                else:
                    mask = (maskImage[:,:,0:1] > 0).astype('float32')
                logger.info("frameId: {} - {}".format(frameId, chunk_image_paths[frameId][0]))

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