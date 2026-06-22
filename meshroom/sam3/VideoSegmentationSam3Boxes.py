__version__ = "4.0"

import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
from pyalicevision import parallelization as avpar

import logging
logger = logging.getLogger("VideoSegmentationSam3Boxes")

class VideoSegmentationSam3Boxes(desc.Node):
    size = avpar.DynamicViewsSize("input")
    gpu = lambda node: desc.Level.EXTREME if node.useOnlyHighPowerGpu.value else desc.Level.INTENSIVE
    # _cuda_tag = "cuda24G"

    category = "Segmentation"
    documentation = """
## Video Segmentation with SAM3 Bounding Boxes

Generates binary segmentation masks for video sequences using **SAM3** (Segment Anything Model 3),
guided by bounding boxes from a `bboxes.json` file (typically produced by **VideoSegmentationSam3Text**).

### Bounding Box Pre-processing
- **Force Squared Boxes**: Converts boxes to squares using the longest side.
- **Box Extension Factor**: Scales each box side by a given factor (e.g. `1.05` adds 5% margin).

### Multi-Resolution Support
When tiling is **disabled**, the resolution used per bounding box is selected automatically:
- Box < **252×252** px → ×4 image (if available)
- Box < **504×504** px → ×2 image (if available)
- Otherwise → native resolution

### Fit SAM3 Model Input Size (`roundCropSize`)
Snaps crop dimensions to **252, 504 or 1008** px to improve model accuracy on small boxes.
Requires **Force Squared Boxes**. When tiling is enabled, applies to all square tiles regardless.

### Tiling Mode
Subdivides large bounding boxes into overlapping tiles before inference.

- **Target Tile Size**: Target size (px) for each tile.
- **Minimal Overlap**: Minimum overlap (px) between adjacent tiles.
- **Maximal Number Of Tiles Per Dimension**: Caps tile count along width and height.
- **Fine Masks Minimal IoU With Coarse Mask** (`minIoU`): A coarse segmentation is first computed
  on the full box. If a fine tile mask IoU with the coarse mask falls below this threshold,
  the coarse mask is used as fallback for that tile.

> When tiling is enabled, native resolution images are always used.

### Tile Size Calculation (`tile_chunk`)

Divides a bounding box into a grid of overlapping tiles. The tile size balances three constraints:
a **target tile size**, a **minimum overlap** and a **maximum tile count per dimension**.

1. **Minimum tile size** respecting max tile count with at least `min_overlap` px between tiles:

        tile_size_min = ceil( (box_size + (N_max - 1) * min_overlap) / N_max )

2. **Effective tile size**:

        tile_size = clamp( max(targetTileSize, tile_size_min), tile_size_min, box_size )

3. **Optional snap** (square box + `roundCropSize` enabled): snaps tile size up to the nearest
   SAM3-compatible threshold (252, 504, 1008) if the box exceeds that threshold.

4. **Tile count and actual overlap**:

        N = floor(box_size / tile_size) + 1
        overlap = floor( (N * tile_size - box_size) / (N - 1) )

   If `overlap < min_overlap`: N += 1 and overlap is recomputed.

5. **Tile positions**: computed with `step = tile_size - overlap`. The last tile is anchored
   to `box_size - tile_size` to guarantee full coverage.

### Processing Pipeline
For each tracked object:
1. Bounding boxes are read from `bboxes.json` and grouped into temporal chunks.
2. *(Tiling only)* A coarse segmentation pass is run on the full box as IoU reference,
   unless pre-computed masks are provided via **Masks Folder**.
3. Each chunk is split into tiles (if tiling enabled).
4. Cropped image sequences are fed to the SAM3 video predictor with a text prompt on the first frame;
   masks are propagated across all frames.
5. Tile masks are resized and composited into full-resolution masks using a **union** operation.
6. *(Tiling only)* Tiles with IoU below `minIoU` are replaced by the coarse mask crop.
7. Masks are saved with optional inversion and bounding box metadata in file headers.

### Debug Mode
When **Verbose Level** is `debug` and **Draw Tiles On Mask In Debug Mode** is enabled,
tile borders are baked into output masks in red for visual inspection.

### Output
Single-channel masks (white = object, black = background, or inverted).
Filenames use the original input name or the view ID depending on **Keep Filename**.
Bounding box metadata is embedded under the `Meshroom:mrSegmentation:` namespace.
"""

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="SfMData file.",
            value="",
        ),
        desc.File(
            name="inputx2",
            label="Input x2",
            description="Folder containing source images upscaled by 2.",
            value="",
        ),
        desc.File(
            name="inputx4",
            label="Input x4",
            description="Folder containing source images upscaled by 4.",
            value="",
        ),
        desc.File(
            name="bboxesFolder",
            label="Bounding Boxes Folder",
            description="Folder containing the bboxes.json file associated to the sfmData used as input.",
            value="",
        ),
        desc.File(
            name="masksFolder",
            label="Masks Folder",
            description="Folder containing already computed masks that can be used as rough masks in case of tiling.\n"
                        "If unset when tiling is enabled, rough masks will be computed in every bounding boxes ",
            value="",
        ),
        desc.BoolParam(
            name="forceSquaredBoxes",
            label="Force Squared Boxes",
            description="Transform rectangle boxes into square ones. The square side is the largest side of the rectangle.",
            value=False,
        ),
        desc.BoolParam(
            name="roundCropSize",
            label="Fit Sam3 Model Input Size If Possible",
            description="Round crop size to 252x252, 504x504 or 1008x1008 for tube with smaller bounding boxes.",
            value=True,
            enabled=lambda node: node.forceSquaredBoxes.value,
        ),
        desc.FloatParam(
            name="boxExtensionFactor",
            label="Box Extension Factor",
            description="Multiply all the box sides by this factor.",
            value=1.05,
            range=(1.0, 2.0, 0.01),
        ),
        desc.BoolParam(
            name="enableTiling",
            label="Enable Tiling",
            description="Enable tiling in big boxes.",
            value=True,
        ),
        desc.IntParam(
            name="targetTileSize",
            label="Target Tile Size",
            description="Tile size.",
            value=504,
            enabled=lambda node: node.enableTiling.value,
        ),
        desc.IntParam(
            name="minimalOverlap",
            label="Minimal Overlap",
            description="Minimal tile overlap.",
            value=16,
            range=(1, 1008, 1),
            enabled=lambda node: node.enableTiling.value,
        ),
        desc.IntParam(
            name="maximalNumberOfTilesPerDimension",
            label="Maximal Number Of Tiles Per Dimension",
            description="Maximal number of tiles for width and height.",
            value=2,
            range=(1, 8, 1),
            enabled=lambda node: node.enableTiling.value,
        ),
        desc.FloatParam(
            name="minIoU",
            label="Fine Masks Minimal IoU With Coarse Mask",
            description="Minimal IoU between coarse and fine mask within a tile to keep the fine mask.",
            value=0.5,
            enabled=lambda node: node.enableTiling.value,
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
            name="useOnlyHighPowerGpu",
            label="Use Only High Power GPU",
            description="Set GPU power requirement.",
            value=True,
            invalidate=False,
        ),
        desc.BoolParam(
            name="computeOnFirstFrameOnly",
            label="Compute On First Frame Only",
            description="Compute segmentation only on the first frame.",
            value=False,
        ),
        desc.BoolParam(
            name="keepFilename",
            label="Keep Filename",
            description="Keep the filename of the inputs for the outputs.",
            value=True,
        ),
        desc.ChoiceParam(
            name="extensionOut",
            label="Output File Extension",
            description="Output image file extension.\n"
                        "If unset, the output file extension will match the input's if possible.",
            value="exr",
            values=["exr", "png", "jpg"],
            exclusive=True,
        ),
        desc.BoolParam(
            name="drawTilesInDebug",
            label="Draw Tiles On Mask In Debug Mode",
            description="Bake tile borders in debug mask images.",
            value=True,
            enabled=lambda node: node.verboseLevel.value == "debug",
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
        ),
    ]

    def resolvePaths(self, node):
        input_path = node.input.value
        image_paths = get_image_paths_list(input_path, node.inputx2.value, node.inputx4.value, node.masksFolder.value)
        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {input_path}')
        if node.computeOnFirstFrameOnly.value:
            self.image_paths = [image_paths[0]]
        else:
            self.image_paths = image_paths
        if node.bboxesFolder.value == "":
            raise ValueError(f'No file containing bounding boxes connected')

    def processChunk(self, chunk):
        from segmentationRDS import image, sam3Utils, bboxUtils
        from sam3.model_builder import build_sam3_video_predictor
        import numpy as np
        import torch
        from pyalicevision import image as avimg
        from PIL import Image
        import OpenImageIO as oiio

        try:

            self.resolvePaths(chunk.node)
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

            gpus_to_use = [torch.cuda.current_device()] if torch.cuda.is_available() else None
            video_predictor = build_sam3_video_predictor(checkpoint_path=chunk.node.segmentationModelPath.evalValue, gpus_to_use=gpus_to_use)

            metadata_deep_model = {}
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelName"] = "SegmentAnything"
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelVersion"] = "sam3-Video-Crop"
            metadata_deep_model["Meshroom:mrSegmentation:NodeVersion"] = "sam3Boxes-" + __version__

            # bboxes.json decoding
            json_path = os.path.join(chunk.node.bboxesFolder.value, "bboxes.json")
            frame_w = chunk_image_paths[0][3]
            frame_h = chunk_image_paths[0][4]
            par = chunk_image_paths[0][5]
            firstFrameId = chunk_image_paths[0][2]
            x2_ok = os.path.exists(chunk.node.inputx2.value)
            x4_ok = os.path.exists(chunk.node.inputx4.value)
            roundCrop = chunk.node.roundCropSize.value
            squareBox = chunk.node.forceSquaredBoxes.value
            exp_factor = chunk.node.boxExtensionFactor.value
            bboxes = bboxUtils.extract_tracking(json_path, frame_w, frame_h, x2_ok, x4_ok, roundCrop, squareBox, exp_factor, par)
            bboxes_metadata = bboxUtils.extract_tracking(json_path, frame_w, frame_h, False, False, roundCrop, squareBox, exp_factor, par)
            metadata_boxes = {}
            for frameId in range(len(chunk_image_paths)):
                metadata_boxes[firstFrameId + frameId] = {}

            logger.debug(f"bboxes.keys() = {bboxes.keys()}")

            full_mask_images = {}
            full_rough_mask_images = {}
            rough_mask_available = chunk_image_paths[0][8] is not None
            img, h_ori, w_ori, p_a_r, orientation = image.loadImage(str(chunk_image_paths[0][0]), True)
            sourceInfo = {"h_ori": h_ori, "w_ori": w_ori, "PAR": p_a_r, "orientation": orientation}
            maskNbChannel = 3 if chunk.node.verboseLevel.value.upper() == "DEBUG" and chunk.node.drawTilesInDebug.value else 1
            for frameId, image_path in enumerate(chunk_image_paths):
                full_mask_images[image_path[2]] = np.zeros((img.shape[0], img.shape[1], maskNbChannel), np.float32)
                if chunk.node.enableTiling.value and not rough_mask_available:
                    full_rough_mask_images[image_path[2]] = np.zeros((img.shape[0], img.shape[1], 1), np.float32)

            for key, frame_chunks in bboxes.items():

                if "_" in key:
                    textPrompt, obj_id = key.rsplit('_', 1)
                else:
                    textPrompt, obj_id = key, ""
                logger.info(f"key = {key} ; text prompt = {textPrompt} ; obj_id = {obj_id}")

                for frame_chunk in frame_chunks:
                    logger.info(f"frame_chunk:\n{frame_chunk}")
                    logger.debug(f"{frame_chunk.boxes}")

                    chunk_tiles = [frame_chunk]
                    if chunk.node.enableTiling.value:
                        chunk_tiles_tmp = bboxUtils.tile_chunk(frame_chunk, chunk.node.targetTileSize.value,
                                                               chunk.node.minimalOverlap.value,
                                                               chunk.node.maximalNumberOfTilesPerDimension.value, sourceInfo["PAR"],
                                                               chunk.node.forceSquaredBoxes.value and chunk.node.roundCropSize.value, logger)
                        for chunk_tile in chunk_tiles_tmp:
                            chunk_tiles.append(chunk_tile)

                    # In tiling mode, avoid loading all frames for every new tiles
                    full_pil_images = {}
                    if chunk.node.enableTiling.value and len(chunk_tiles) > 1:
                        pil_images = []
                        for frameId, box in chunk_tiles[0].boxes.items():
                            if not chunk.node.computeOnFirstFrameOnly.value or frameId == chunk_image_paths[0][2]:
                                img, h_ori, w_ori, PAR, orientation = image.loadImage(str(chunk_image_paths[frameId - firstFrameId][0]), True)
                                full_pil_images[frameId] = img
                                x1, y1, x2, y2 = bboxUtils.box_to_display(box, sourceInfo["PAR"])
                                imgBuf = oiio.ImageBuf(img)
                                imgBuf = oiio.ImageBufAlgo.crop(imgBuf, roi=oiio.ROI(x1, x2, y1, y2))
                                img_crop = imgBuf.get_pixels(format=oiio.FLOAT)
                                pil_images.append(Image.fromarray((255.0*img_crop).astype("uint8")))
    
                        if not rough_mask_available:
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
                                    text=textPrompt,
                                )
                            )
                            outputs_per_frame = sam3Utils.propagateInVideo(video_predictor, session_id)
                            outputs_per_frame_visu = sam3Utils.prepareMasksForVisualization(outputs_per_frame)

                            for frame_idx, box in sorted(chunk_tiles[0].boxes.items()):
                                if not chunk.node.computeOnFirstFrameOnly.value or frame_idx == chunk_image_paths[0][2]:
                                    x1, y1, x2, y2 = bboxUtils.box_to_display(box, sourceInfo["PAR"])
                                    box_w = x2 - x1
                                    box_h = y2 - y1
                                    tgt = full_rough_mask_images[frame_idx][y1:y2 ,x1:x2, :]
                                    mask_in_full_box = np.zeros_like(tgt)
                                    frameId = frame_idx - chunk_tiles[0].start_frame
                                    for key, maskBoxProb in outputs_per_frame_visu[frameId].items():
                                        mask = maskBoxProb["mask"]
                                        buf_in = oiio.ImageBuf(mask.astype('float32'))
                                        buf_out = oiio.ImageBufAlgo.resample(buf_in, roi=oiio.ROI(0, box_w, 0, box_h))
                                        mask = buf_out.get_pixels().reshape(box_h, box_w, 1)
                                        bool_mask = mask.squeeze() > 0
                                        mask_in_full_box[bool_mask] = [1.0]
                                    full_rough_mask_images[frame_idx][y1:y2, x1:x2, :] = mask_in_full_box

                        chunk_tiles.pop(0)

                    logger.info(f"chunk_tiles:\n{chunk_tiles}")

                    for tile_idx, chunk_tile in enumerate(chunk_tiles):
                        logger.debug(f"Tile {tile_idx+1}/{len(chunk_tiles)}: {chunk_tile.boxes}")

                        pil_images = []
                        for frame_idx, box in sorted(chunk_tile.boxes.items()):

                            if not chunk.node.computeOnFirstFrameOnly.value or frame_idx == chunk_image_paths[0][2]:
                                x1, y1, x2, y2 = bboxUtils.box_to_display(box, sourceInfo["PAR"])
                                box_w = x2 - x1
                                box_h = y2 - y1

                                if box_w <= 252 and box_h <= 252 and x4_ok and not chunk.node.enableTiling.value:
                                    img, h_ori, w_ori, p_a_r, orientation = image.loadImage(str(chunk_image_paths[frame_idx - firstFrameId][7]), True)
                                    imgBuf = oiio.ImageBuf(img)
                                    imgBuf = oiio.ImageBufAlgo.crop(imgBuf, roi=oiio.ROI(4*x1, 4*x2, 4*y1, 4*y2))
                                elif box_w <= 504 and box_h <= 504 and x2_ok and not chunk.node.enableTiling.value:
                                    img, h_ori, w_ori, p_a_r, orientation = image.loadImage(str(chunk_image_paths[frame_idx - firstFrameId][6]), True)
                                    imgBuf = oiio.ImageBuf(img)
                                    imgBuf = oiio.ImageBufAlgo.crop(imgBuf, roi=oiio.ROI(2*x1, 2*x2, 2*y1, 2*y2))
                                elif not chunk.node.enableTiling.value:
                                    img, h_ori, w_ori, p_a_r, orientation = image.loadImage(str(chunk_image_paths[frame_idx - firstFrameId][0]), True)
                                    imgBuf = oiio.ImageBuf(img)
                                    imgBuf = oiio.ImageBufAlgo.crop(imgBuf, roi=oiio.ROI(x1, x2, y1, y2))
                                else:
                                    # use already loaded images
                                    imgBuf = oiio.ImageBuf(full_pil_images[frame_idx])
                                    imgBuf = oiio.ImageBufAlgo.crop(imgBuf, roi=oiio.ROI(x1, x2, y1, y2))

                                img_crop = imgBuf.get_pixels(format=oiio.FLOAT)
                                pil_images.append(Image.fromarray((255.0*img_crop).astype("uint8")))

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
                                text=textPrompt,
                            )
                        )
                        outputs_per_frame = sam3Utils.propagateInVideo(video_predictor, session_id)
                        outputs_per_frame_visu = sam3Utils.prepareMasksForVisualization(outputs_per_frame)

                        for frame_idx, box in sorted(chunk_tile.boxes.items()):
                            if not chunk.node.computeOnFirstFrameOnly.value or frame_idx == chunk_image_paths[0][2]:
                                x1, y1, x2, y2 = bboxUtils.box_to_display(box, sourceInfo["PAR"])
                                box_w = x2 - x1
                                box_h = y2 - y1
                                tgt = full_mask_images[frame_idx][y1:y2 ,x1:x2, :]
                                fine_mask = np.zeros_like(tgt)
                                frameId = frame_idx - chunk_tile.start_frame
                                logger.debug(f"frame: {frame_idx}; tile: {box}; items number: {len(outputs_per_frame_visu[frameId].keys())}")
                                for key, maskBoxProb in outputs_per_frame_visu[frameId].items():
                                    mask = maskBoxProb["mask"]
                                    buf_in = oiio.ImageBuf(mask.astype('float32'))
                                    buf_out = oiio.ImageBufAlgo.resample(buf_in, roi=oiio.ROI(0, box_w, 0, box_h))
                                    mask = buf_out.get_pixels().reshape(box_h, box_w, 1)
                                    bool_mask = mask.squeeze() > 0
                                    fine_mask[bool_mask] = [255, 255, 255] if maskNbChannel == 3 else [255]

                                if chunk.node.enableTiling.value:
                                    if frame_idx not in full_rough_mask_images and rough_mask_available:
                                        if os.path.isfile(str(chunk_image_paths[frame_idx - firstFrameId][8])):
                                            logger.info(f"read mask for frame {frame_idx} at {chunk_image_paths[frame_idx - firstFrameId][8]}")
                                            maskImg, h_mask, w_mask, PAR_mask, or_mask = image.loadImage(str(chunk_image_paths[frame_idx - firstFrameId][8]), True)
                                            x1, y1, x2, y2 = bboxUtils.box_to_display(box, PAR_mask)
                                            imgBuf = oiio.ImageBuf(maskImg)
                                            imgBuf = oiio.ImageBufAlgo.crop(imgBuf, roi=oiio.ROI(x1, x2, y1, y2))
                                            full_rough_mask_images[frame_idx] = np.zeros((maskImg.shape[0], maskImg.shape[1], 1), np.float32)
                                            full_rough_mask_images[frame_idx][y1:y2, x1:x2, :] = imgBuf.get_pixels(format=oiio.FLOAT)[:, :, 0:1]

                                    if frame_idx in full_rough_mask_images:
                                        imgBuf = oiio.ImageBuf(full_rough_mask_images[frame_idx])
                                        imgBuf = oiio.ImageBufAlgo.crop(imgBuf, roi=oiio.ROI(x1, x2, y1, y2))
                                        rough_mask_crop = imgBuf.get_pixels(format=oiio.FLOAT)
                                        if rough_mask_crop.ndim == 3 and rough_mask_crop.shape[2] >= 3:
                                            rough_mask_R = rough_mask_crop[:, :, 0]
                                        else:
                                            rough_mask_R = rough_mask_crop.squeeze()

                                        intersection = np.logical_and(fine_mask[:, :, 0], rough_mask_R).sum()
                                        union = np.logical_or(fine_mask[:, :, 0], rough_mask_R).sum()
                                        if union == 0:
                                            IoU = 0.0
                                        else:
                                            IoU = float(intersection / union)

                                        logger.debug(f"IoU = {IoU}")
                                        if IoU < chunk.node.minIoU.value:
                                            logger.info(f"frame_idx = {frame_idx}; Tile = {tile_idx}; IoU = {IoU} => Use rough mask for this tile")
                                            fine_mask = rough_mask_crop

                                        if rough_mask_available:
                                            del full_rough_mask_images[frame_idx]

                                # Union where tiles overlap
                                existing_mask = full_mask_images[frame_idx][y1:y2 ,x1:x2, 0]
                                final_mask = np.logical_or(fine_mask[:, :, 0], existing_mask)
                                full_mask_images[frame_idx][y1:y2, x1:x2, 0:1] = np.dstack([final_mask])
                                if chunk.node.verboseLevel.value.upper() == "DEBUG" and chunk.node.drawTilesInDebug.value:
                                    full_mask_images[frame_idx][y1:y2, x1:x1+1, 1:3] = [255, 255]
                                    full_mask_images[frame_idx][y1:y2, x2:x2+1, 1:3] = [255, 255]
                                    full_mask_images[frame_idx][y1:y1+1, x1:x2, 1:3] = [255, 255]
                                    full_mask_images[frame_idx][y2:y2+1, x1:x2, 1:3] = [255, 255]

                        video_predictor.handle_request(request=dict(type="close_session", session_id=session_id))

            for key, frame_chunks in bboxes_metadata.items():
                if "_" in key:
                    textPrompt, obj_id = key.rsplit('_', 1)
                else:
                    textPrompt, obj_id = key, ""
                for frame_chunk in frame_chunks:
                    for frame_idx, box in sorted(frame_chunk.boxes.items()):
                        if not chunk.node.computeOnFirstFrameOnly.value or frame_idx == chunk_image_paths[0][2]:
                            if textPrompt not in metadata_boxes[frame_idx]:
                                metadata_boxes[frame_idx][textPrompt] = {}
                            x1, y1, x2, y2 = box
                            bbox_str = str(x1) + ";" + str(y1)+ ";" + str(x2)+ ";" + str(y2)
                            metadata_boxes[frame_idx][textPrompt][textPrompt + "_" + str(obj_id)] = bbox_str

            for frameId, image_path in enumerate(chunk_image_paths):
                m = full_mask_images[image_path[2]][:,:,0:1] > 0
                if chunk.node.maskInvert.value:
                    mask = np.ones_like(full_mask_images[image_path[2]])
                    mask[m[:, :, 0]] = [0, 0, 0] if maskNbChannel == 3 else [0]
                else:
                    mask = np.zeros_like(full_mask_images[image_path[2]])
                    mask[m[:, :, 0]] = [1.0, 1.0, 1.0] if maskNbChannel == 3 else [1.0]
                if chunk.node.verboseLevel.value.upper() == "DEBUG" and chunk.node.drawTilesInDebug.value:
                    g = full_mask_images[image_path[2]][:,:,1:2] > 0
                    mask[g[:, :, 0]] = [1.0, 0, 0]
                logger.info(f"frameId: {frameId} - {image_path[0]}")

                if chunk.node.keepFilename.value:
                    outputFileMask = os.path.join(chunk.node.output.value, Path(image_path[0]).stem + "." + chunk.node.extensionOut.value)
                else:
                    outputFileMask = os.path.join(chunk.node.output.value, str(image_path[1]) + "." + chunk.node.extensionOut.value)

                optWrite = avimg.ImageWriteOptions()
                optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
                if Path(outputFileMask).suffix.lower() == ".exr":
                    optWrite.exrCompressionMethod(avimg.EImageExrCompression_stringToEnum("DWAA"))
                    optWrite.exrCompressionLevel(300)

                frame_metadata_deep_model = dict(metadata_deep_model)
                for prompt, bboxes in metadata_boxes[firstFrameId + frameId].items():
                    for k, box in metadata_boxes[firstFrameId + frameId][prompt].items():
                            frame_metadata_deep_model["Meshroom:mrSegmentation:" + k] = box

                image.writeImage(outputFileMask, mask, sourceInfo["h_ori"], sourceInfo["w_ori"], sourceInfo["orientation"],
                                 sourceInfo["PAR"], frame_metadata_deep_model, optWrite)

        finally:
            torch.cuda.empty_cache()


def get_image_paths_list(input_path, path_folder_x2 = "", path_folder_x4 = "", mask_folder = ""):
    from pyalicevision import sfmData, camera
    from pyalicevision import sfmDataIO
    from pathlib import Path

    image_paths = []

    if Path(input_path).suffix.lower() in [".sfm", ".abc"]:
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                commonParams = None
                for id, v in views.items():
                    image_x1_path = Path(v.getImage().getImagePath())
                    image_x1_name = image_x1_path.name
                    image_x2_path = None
                    if os.path.isfile(os.path.join(path_folder_x2, image_x1_name)):
                        image_x2_path = os.path.join(path_folder_x2, image_x1_name)
                    image_x4_path = None
                    if os.path.isfile(os.path.join(path_folder_x4, image_x1_name)):
                        image_x4_path = os.path.join(path_folder_x4, image_x1_name)
                    mask_path = None
                    if os.path.isfile(os.path.join(mask_folder, image_x1_name)):
                        mask_path = os.path.join(mask_folder, image_x1_name)
                    intrinsic = dataAV.getIntrinsicSharedPtr(v.getIntrinsicId())
                    pinhole = camera.Pinhole.cast(intrinsic)
                    par = 1.0
                    if pinhole is not None:
                        par = pinhole.getPixelAspectRatio()
                    params = [v.getImage().getWidth(), v.getImage().getHeight(), par, image_x2_path is None, image_x4_path is None]
                    if commonParams is None:
                        commonParams = params
                    if commonParams != params:
                        raise ValueError(f"All images do not have same dimensions or one image is missing its upscaled version: {params} vs {commonParams}")
                    image_paths.append((image_x1_path, str(id), v.getFrameId(), v.getImage().getWidth(),
                                        v.getImage().getHeight(), par, image_x2_path, image_x4_path, mask_path))

            image_paths.sort(key=lambda x: x[0])
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid sfmData file.")
    return image_paths