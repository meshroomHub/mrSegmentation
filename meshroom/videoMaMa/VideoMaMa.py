__version__ = "1.0"

from distutils.log import error
from functools import total_ordering
import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
from pyalicevision import parallelization as avpar

import logging
logger = logging.getLogger("VideoMaMa")

class VideoMaMa(desc.Node):
    """
## Matting with VideoMaMa
"""
    size = avpar.DynamicViewsSize("input")
    gpu = lambda node: desc.Level.EXTREME if node.inferenceSize.value == 2048 else desc.Level.INTENSIVE
    _cuda_tag = "cuda24G"

    category = "Matting"

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="SfMData file.",
            value="",
        ),
        desc.File(
            name="inputMask",
            label="Mask Folder",
            description="Folder containing the masks used as prompt.",
            value="",
        ),
        desc.ChoiceParam(
            name="extensionMask",
            label="Mask File Extension",
            description="Input mask file extension.",
            value="exr",
            values=["exr", "png", "jpg"],
            exclusive=True,
        ),
        desc.ChoiceParam(
            name="inferenceSize",
            label="Inference Size Max",
            description="Maximum size of the largest image dimension for inference. Automatic resize if higher.",
            value=1024,
            values=[512, 640, 768, 896, 1024, 1576, 2048],
            exclusive=True,
        ),
        desc.IntParam(
            name="batchSize",
            label="Batch Size",
            description="Number of frames process simultaneously.",
            value=16,
        ),
        desc.IntParam(
            name="overlap",
            label="Overlap",
            description="Number of overlaping frames between 2 consecutive batches. Must be lower than batch size.",
            value=2,
        ),
        desc.FloatParam(
            name="boxExtensionFactor",
            label="Bounding Box Extension Factor",
            description="Extension factor of bounding boxes containing binary masks.",
            value=1.1,
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
            name="extensionOut",
            label="Output File Extension",
            description="Output image file extension.\n"
                        "If unset, the output file extension will match the input's if possible.",
            value="exr",
            values=["exr", "png", "jpg"],
            exclusive=True,
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
            label="Mattes Folder",
            description="Output path for the mattes.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="matte",
            label="Matte",
            description="Generated mattes.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + "." + attr.node.extensionOut.value,
        ),
    ]

    def onInferenceSizeChanged(self, node):
        if node.inferenceSize.value < 2048:
            _cuda_tag = "cuda24G"
        else:
            _cuda_tag = ""

    def _resolve_paths(self, pathIn, pathMask, extMask, outDir, keepFilename, extOut):
        from pyalicevision import sfmData, camera
        from pyalicevision import sfmDataIO
        from pathlib import Path

        paths = []
        inputFileMask = None
        if not Path(pathIn).exists():
            raise FileNotFoundError(f"Input path '{pathIn}' does not exist.")
        if not Path(pathIn).suffix.lower() in [".sfm", ".abc"]:
            raise ValueError(f"Input path '{pathIn}' is not a valid sfmData file.")
        if not os.path.exists(os.path.join(pathMask,"bboxes.json")):
            raise FileNotFoundError(f'No file containing bounding boxes')
        
        dataAV = sfmData.SfMData()
        if sfmDataIO.load(dataAV, pathIn, sfmDataIO.ALL) and os.path.isdir(outDir):
            views = dataAV.getViews()
            for id, v in views.items():
                inputFile = v.getImage().getImagePath()
                frameId = v.getFrameId()
                imgWidth = v.getImage().getWidth()
                imgHeight = v.getImage().getHeight()
                intrinsic = dataAV.getIntrinsicSharedPtr(v.getIntrinsicId())
                pinhole = camera.Pinhole.cast(intrinsic)
                par = 1.0
                if pinhole is not None:
                    par = pinhole.getPixelAspectRatio()
                if keepFilename:
                    if pathMask:
                        inputFileMask = os.path.join(pathMask, Path(inputFile).stem + "." + extMask)
                    outputFileMatte = os.path.join(outDir, Path(inputFile).stem + "." + extOut)
                else:
                    if pathMask:
                        inputFileMask = os.path.join(pathMask, str(id) + "." + extMask)
                    outputFileMatte = os.path.join(outDir, str(id) + "." + extOut)
                paths.append((inputFile, inputFileMask, frameId, str(id), outputFileMatte, imgWidth, imgHeight, par))
            paths.sort(key=lambda x: x[0])

        return paths

    def _resize_image(self, image, max_size):
        """Resize image and ensure dimensions are multiples of 8 for the model."""
        import cv2

        h, w = image.shape[:2]
        
        scale = 1.0
        if max_size > 0:
            max_side = max(h, w)
            if max_side > max_size:
                scale = max_size / max_side

        new_h = (int(h * scale) // 8) * 8
        new_w = (int(w * scale) // 8) * 8
        
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _restore_image_size(self, image, original_size):
        """Restore image to original size. original_size must be (w, h) as expected by cv2."""
        import cv2
        original_w, original_h = original_size
        restored_image = cv2.resize(image, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        return restored_image

    def _generate_time_slices(self, total_frames, batch_size, overlap):
        step = batch_size - overlap
        if step <= 0:
            overlap = batch_size - 1
            step = 1

        if total_frames <= batch_size:
            return [(0, total_frames)]

        time_slices = []
        pos = 0
        while pos < total_frames:
            end = min(pos + batch_size, total_frames)
            time_slices.append((pos, end))
            if end >= total_frames:
                break
            pos += step

        return time_slices

    def _check_lists_compatibility(self, list1, list2):
        if len(list1) != len(list2):
            raise ValueError("Lists have different size.")
        if len(list1) == 0:
            return 0, None
        for (arr1, arr2) in zip(list1, list2):
            if arr1.shape != arr2.shape:
                raise ValueError("List content don't match.")
        return len(list1), list1[0].shape

    def processChunk(self, chunk):
        from segmentationRDS import image, bboxUtils, videoMaMaUtils

        import cv2
        import numpy as np
        import torch
        from pyalicevision import image as avimg
        import OpenImageIO as oiio

        try:
            logger.setLevel(chunk.node.verboseLevel.value.upper())

            if not chunk.node.input:
                logger.warning("Nothing to segment.")
                return

            logger.info("Chunk range from {} to {}".format(chunk.range.start, chunk.range.last))

            chunk_image_paths = self._resolve_paths(chunk.node.input.value,
                                                    chunk.node.inputMask.value, chunk.node.extensionMask.value,
                                                    chunk.node.output.value, chunk.node.keepFilename.value,
                                                    chunk.node.extensionOut.value)

            if not os.path.exists(chunk.node.output.value):
                os.mkdir(chunk.node.output.value)

            device = torch.device("cuda") if torch.cuda.is_available() and chunk.node.useGpu.value else torch.device("cpu")
            model_path = os.path.join(os.getenv("VIDEOMAMA_MODELS_PATH"), "fromSammieRoto")

            try:
                pipeline = videoMaMaUtils.VideoInferencePipeline(
                    base_model_path=model_path,
                    unet_checkpoint_path=model_path,
                    weight_dtype=torch.float16,
                    device=str(device),
                    enable_model_cpu_offload=False, # Not much benefit here, since the vae is a small model
                    vae_encode_chunk_size=1,        # Process VAE in small chunks, increasing doesnt help anything
                    attention_mode="auto",          # Use xformers if available, else SDPA
                    enable_vae_tiling=False,        # Tiling VAE is not worth it
                    enable_vae_slicing=True,        # Process VAE one image at a time
                )
                print(f"Loaded videoMaMa model to {device}") # with max size {max_size}, overlap={overlap}, batch={batch_size}, and combined={combined}")
            except Exception as e:
                raise ValueError(f"Error loading VideoMaMa pipeline: {e}")

            metadata_deep_model_base = {
            "Meshroom:mrSegmentation:DeepModelName": "VideoMaMa",
            "Meshroom:mrSegmentation:DeepModelVersion": "0.1",
            "Meshroom:mrSegmentation:NodeVersion": "VideoMaMa-" + __version__
            }

            # bboxes.json decoding
            json_path = os.path.join(chunk.node.inputMask.value, "bboxes.json")
            frame_w = chunk_image_paths[0][5]
            frame_h = chunk_image_paths[0][6]
            par = chunk_image_paths[0][7]
            firstFrameId = chunk_image_paths[0][2]
            exp_factor = chunk.node.boxExtensionFactor.value
            bboxes = bboxUtils.extract_tracking(json_path, frame_w, frame_h, False, False, False, False, exp_factor, par)
            bboxes_metadata = bboxUtils.extract_tracking(json_path, frame_w, frame_h, False, False, False, False, exp_factor, par)
            metadata_boxes = {}
            for frameId in range(len(chunk_image_paths)):
                metadata_boxes[firstFrameId + frameId] = {}

            logger.debug(f"bboxes.keys() = {bboxes.keys()}")

            full_alpha = {}
            img, h_ori, w_ori, p_a_r, orientation = image.loadImage(str(chunk_image_paths[0][0]), True)
            sourceInfo = {"h_ori": h_ori, "w_ori": w_ori, "PAR": p_a_r, "orientation": orientation}
            for frameId, image_path in enumerate(chunk_image_paths):
                full_alpha[image_path[2]] = np.zeros_like(img)

            batch_size = chunk.node.batchSize.value
            overlap = chunk.node.overlap.value

            for key, frame_chunks in bboxes.items():

                if "_" in key:
                    textPrompt, obj_id = key.rsplit('_', 1)
                else:
                    textPrompt, obj_id = key, ""
                logger.info(f"key = {key} ; text prompt = {textPrompt} ; obj_id = {obj_id}")

                for frame_chunk in frame_chunks:
                    logger.info(f"frame_chunk:\n{frame_chunk}")
                    logger.debug(f"{frame_chunk.boxes}")

                    total_frames = frame_chunk.end_frame - frame_chunk.start_frame + 1
                    time_slices = self._generate_time_slices(total_frames, batch_size, overlap)
                    logger.debug(f"time_slices = {time_slices}")

                    cond_frames = []
                    mask_frames = []
                    for slice_idx, (slice_start, slice_end) in enumerate(time_slices):
                        # startFrameId = firstFrameId + slice_start
                        # stopFrameId = firstFrameId + slice_end
                        startFrameId = frame_chunk.start_frame + slice_start
                        stopFrameId = frame_chunk.start_frame + slice_end
                        logger.info(f"slice #{slice_idx}/{len(time_slices)-1}: processing frames [{startFrameId}, {stopFrameId}[")
                        if slice_idx > 0:
                            cond_frames = cond_frames[-overlap:]
                            mask_frames = mask_frames[-overlap:]
                            startFrameId += overlap
                        for frameId, box in frame_chunk.boxes.items():
                            if frameId >= startFrameId and frameId < stopFrameId:
                                #img, h_ori, w_ori, PAR, orientation = image.loadImage(str(chunk_image_paths[frameId - firstFrameId][0]), True)
                                img, h_ori, w_ori, PAR, orientation = image.loadImage(str(chunk_image_paths[frameId - frame_chunk.start_frame][0]), True)
                                x1, y1, x2, y2 = bboxUtils.box_to_display(box, sourceInfo["PAR"])
                                imgBuf = oiio.ImageBuf(img)
                                imgBuf = oiio.ImageBufAlgo.crop(imgBuf, roi=oiio.ROI(x1, x2, y1, y2))
                                img_crop = imgBuf.get_pixels(format=oiio.FLOAT)
                                frame = self._resize_image(img_crop, max(chunk.node.inferenceSize.value, 2160))
                                resized_h, resized_w = frame.shape[:2]
                                #mask_path = str(chunk_image_paths[frameId - firstFrameId][1])
                                mask_path = str(chunk_image_paths[frameId - frame_chunk.start_frame][1])
                                mask, h_ori_mask, w_ori_mask, PAR_mask, orientation_mask = image.loadImage(mask_path, True)
                                imgBuf = oiio.ImageBuf(mask)
                                imgBuf = oiio.ImageBufAlgo.crop(imgBuf, roi=oiio.ROI(x1, x2, y1, y2))
                                img_crop = imgBuf.get_pixels(format=oiio.FLOAT)
                                mask = cv2.resize(img_crop, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
                                cond_frames.append(frame)
                                mask_frames.append(mask)
                        nb, sh = self._check_lists_compatibility(cond_frames, mask_frames)
                        logger.info(f"slice_idx = {slice_idx} ; {nb} frames ; shape = {sh}")

                        try:
                            with torch.amp.autocast('cuda', enabled=False):
                                output_frames = pipeline.run(cond_frames=cond_frames, mask_frames=mask_frames, seed=42)
                        except Exception as ex:
                            logger.error(f"Error in VideoMaMa inference at slice {slice_idx}: {ex}")
                            raise

                        if len(output_frames) >= overlap and slice_idx < len(time_slices) - 1:
                            previous_frames = output_frames[-overlap:].copy()
                        if slice_idx == 0:
                            mix_frames = output_frames[0:overlap]
                        else:
                            mix_frames = []
                            for i in range(overlap):
                                new_weight = (i + 1) / (overlap + 1)
                                blended_frame = (1.0 - new_weight) * previous_frames[i] + new_weight * output_frames[i].copy()
                                mix_frames.append(blended_frame)

                        if slice_idx > 0:
                            startFrameId -= overlap
                        if slice_idx < len(time_slices) - 1:
                            stopFrameId -= overlap

                        for frameId, box in sorted(frame_chunk.boxes.items()):
                            if frameId >= startFrameId and frameId < stopFrameId:
                                frame_idx = frameId - startFrameId
                                if frame_idx < batch_size - overlap or slice_idx == len(time_slices) - 1:
                                    x1, y1, x2, y2 = bboxUtils.box_to_display(box, sourceInfo["PAR"])
                                    box_w = x2 - x1
                                    box_h = y2 - y1
                                    output_frame = mix_frames[frame_idx].copy() if frame_idx < overlap else output_frames[frame_idx].copy()
                                    alpha = self._restore_image_size(output_frame, (box_w, box_h))
                                    full_alpha[frameId][y1:y2, x1:x2, :] = alpha

            for key, frame_chunks in bboxes_metadata.items():
                if "_" in key:
                    textPrompt, obj_id = key.rsplit('_', 1)
                else:
                    textPrompt, obj_id = key, ""
                for frame_chunk in frame_chunks:
                    for frame_idx, box in sorted(frame_chunk.boxes.items()):
                        if textPrompt not in metadata_boxes[frame_idx]:
                            metadata_boxes[frame_idx][textPrompt] = {}
                        x1, y1, x2, y2 = box
                        bbox_str = str(x1) + ";" + str(y1)+ ";" + str(x2)+ ";" + str(y2)
                        metadata_boxes[frame_idx][textPrompt][textPrompt + "_" + str(obj_id)] = bbox_str

            for frameId, image_path in enumerate(chunk_image_paths):

                optWrite = avimg.ImageWriteOptions()
                optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
                if Path(image_path[4]).suffix.lower() == ".exr":
                    optWrite.exrCompressionMethod(avimg.EImageExrCompression_stringToEnum("DWAA"))
                    optWrite.exrCompressionLevel(300)

                frame_metadata_deep_model = dict(metadata_deep_model_base)
                for prompt, bboxes in metadata_boxes[firstFrameId + frameId].items():
                    for k, box in metadata_boxes[firstFrameId + frameId][prompt].items():
                            frame_metadata_deep_model["Meshroom:mrSegmentation:" + k] = box
                alpha = full_alpha[image_path[2]]
                image.writeImage(image_path[4], alpha, sourceInfo["h_ori"], sourceInfo["w_ori"], sourceInfo["orientation"],
                                sourceInfo["PAR"], frame_metadata_deep_model, optWrite)

        finally:
            torch.cuda.empty_cache()
