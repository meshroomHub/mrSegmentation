__version__ = "1.0"

import logging
import os
from pathlib import Path

from pyalicevision import parallelization as avpar
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

logger = logging.getLogger("VideoMatting")


class VideoMatting(desc.Node):
    """
Matting node for video sequences.
"""
    size = avpar.DynamicViewsSize("input")
    gpu = lambda node: desc.Level.EXTREME if node.inferenceSize.value == 2048 else desc.Level.INTENSIVE

    category = "Matting"

    inputs = [
        desc.File(
            name="input",
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
            description="Number of frames process simultaneously.",
            value=16,
        ),
        desc.IntParam(
            name="overlap",
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
            description="Generated mattes.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + "." + attr.node.extensionOut.value,
        ),
    ]

    def _resolve_paths(self, input_path, mask_path, mask_ext, output_dir, keep_filename, output_ext):
        from pyalicevision import sfmData, camera
        from pyalicevision import sfmDataIO

        paths = []
        input_file_mask = None
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input path '{input_path}' does not exist.")
        if not Path(mask_path).exists():
            raise FileNotFoundError(f"Input path for masks '{mask_path}' does not exist.")
        if Path(input_path).suffix.lower() not in [".sfm", ".abc"]:
            raise ValueError(f"Input path '{input_path}' is not a valid sfmData file.")
        if not os.path.exists(os.path.join(mask_path,"bboxes.json")):
            raise FileNotFoundError("No file containing bounding boxes.")

        av_data = sfmData.SfMData()
        if sfmDataIO.load(av_data, input_path, sfmDataIO.ALL) and os.path.isdir(output_dir):
            views = av_data.getViews()
            for view_id, view in views.items():
                input_file = view.getImage().getImagePath()
                frame_id = view.getFrameId()
                img_width = view.getImage().getWidth()
                img_height = view.getImage().getHeight()
                intrinsic = av_data.getIntrinsicSharedPtr(view.getIntrinsicId())
                pinhole = camera.Pinhole.cast(intrinsic)
                par = 1.0
                if pinhole is not None:
                    par = pinhole.getPixelAspectRatio()
                if keep_filename:
                    filename = Path(input_file).stem
                    if mask_path:
                        mask_filename = "colorMask_%PROMPT%_merged_" + str(filename)
                        input_file_mask = os.path.join(mask_path, mask_filename + "." + mask_ext)
                    output_file_matte = os.path.join(output_dir, filename + "." + output_ext)
                else:
                    if mask_path:
                        input_file_mask = os.path.join(mask_path, str(view_id) + "." + mask_ext)
                    output_file_matte = os.path.join(output_dir, str(view_id) + "." + output_ext)
                paths.append((input_file, input_file_mask, frame_id, str(view_id), output_file_matte,
                              img_width, img_height, par))
            paths.sort(key=lambda x: x[0])

        return paths

    def _padx8_image(self, image):
        import numpy as np

        height, width = image.shape[:2]
        new_height = (height + 7) // 8 * 8
        new_width = (width + 7) // 8 * 8
        pad_height = new_height - height
        pad_width = new_width - width
        if pad_height == 0 and pad_width == 0:
            return image
        if image.ndim == 2:
            return np.pad(image, ((0, pad_height), (0, pad_width)), mode="constant", constant_values=0)
        return np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant", constant_values=0)

    def _resize_image(self, image, max_size):
        import cv2

        height, width = image.shape[:2]
        scale = 1.0
        if max_size > 0:
            max_side = max(height, width)
            if max_side > max_size:
                scale = max_size / max_side

        if scale < 1.0:
            new_height = (int(height * scale) // 8) * 8
            new_width = (int(width * scale) // 8) * 8
            return "resize", cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return "pad", self._padx8_image(image)

    def _restore_image_size(self, image, original_size, method):
        import cv2

        original_width, original_height = original_size
        if method == "resize":
            restored_image = cv2.resize(image, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        else:
            restored_image = image[0:original_height, 0:original_width, :]
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
                raise ValueError("List content doesn't match.")
        return len(list1), list1[0].shape

    def processChunk(self, chunk):
        from segmentationRDS import image, bboxUtils, videoMattingUtils

        import copy
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
            model_path = os.getenv("VIDEOMATTING_SR_MODELS_PATH")
            if not model_path:
                raise EnvironmentError("VIDEOMATTING_SR_MODELS_PATH is not set; it must point to the folder containing the same VideoMatting model files as the ones used in SammieRoto2.")

            try:
                pipeline = videoMattingUtils.VideoInferencePipeline(
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
                logger.info(f"Loaded VideoMatting model to {device}.")
            except Exception as err:
                raise ValueError(f"Error loading VideoMatting pipeline: {err}.")

            metadata_deep_model_base = {
                "Meshroom:mrSegmentation:DeepModelName": "VideoMatting",
                "Meshroom:mrSegmentation:DeepModelVersion": "0.1",
                "Meshroom:mrSegmentation:NodeVersion": "VideoMatting-" + __version__
            }

            # bboxes.json decoding
            json_path = os.path.join(chunk.node.inputMask.value, "bboxes.json")
            frame_w = chunk_image_paths[0][5]
            frame_h = chunk_image_paths[0][6]
            par = chunk_image_paths[0][7]
            first_frame_id = chunk_image_paths[0][2]
            exp_factor = chunk.node.boxExtensionFactor.value
            bboxes = bboxUtils.extract_tracking(json_path, frame_w, frame_h, False, False, False,
                                                False, exp_factor, par)
            bboxes_metadata = bboxUtils.extract_tracking(json_path, frame_w, frame_h, False, False, False,
                                                         False, exp_factor, par)
            metadata_boxes = {}
            for frame_id in range(len(chunk_image_paths)):
                metadata_boxes[first_frame_id + frame_id] = {}

            logger.debug(f"bboxes.keys() = {bboxes.keys()}")
            prompts = [key.rsplit('_', 1)[0] for key in bboxes.keys()]
            metadata_deep_model_base["Meshroom:mrSegmentation:Prompt"] = ";".join(list(dict.fromkeys(prompts)))

            full_alpha = {}
            img, h_ori, w_ori, p_a_r, orientation = image.loadImage(str(chunk_image_paths[0][0]), True)
            source_info = {"h_ori": h_ori, "w_ori": w_ori, "PAR": p_a_r, "orientation": orientation}
            for frame_id, image_path in enumerate(chunk_image_paths):
                full_alpha[image_path[2]] = np.zeros_like(img)

            batch_size = chunk.node.batchSize.value
            overlap = chunk.node.overlap.value

            color_palette = image.paletteGenerator()

            for key, frame_chunks in bboxes.items():
                if "_" in key:
                    text_prompt, obj_id = key.rsplit('_', 1)
                else:
                    text_prompt, obj_id = key, ""
                logger.info(f"key = {key} ; text prompt = {text_prompt} ; obj_id = {obj_id}")

                for frame_chunk in frame_chunks:
                    logger.info(f"frame_chunk:\n{frame_chunk}")
                    logger.debug(f"{frame_chunk.boxes}")

                    total_frames = frame_chunk.end_frame - frame_chunk.start_frame + 1
                    time_slices = self._generate_time_slices(total_frames, batch_size, overlap)
                    logger.debug(f"time_slices = {time_slices}")

                    cond_frames = []
                    mask_frames = []
                    for slice_idx, (slice_start, slice_end) in enumerate(time_slices):
                        start_frame_id = frame_chunk.start_frame + slice_start
                        stop_frame_id = frame_chunk.start_frame + slice_end
                        logger.info(f"slice #{slice_idx}/{len(time_slices)-1}: processing frames [{start_frame_id}, {stop_frame_id}[")
                        if slice_idx > 0:
                            if overlap > 0:
                                cond_frames = cond_frames[-overlap:]
                                mask_frames = mask_frames[-overlap:]
                                start_frame_id += overlap
                            else:
                                cond_frames = []
                                mask_frames = []
                        for frame_id, box in frame_chunk.boxes.items():
                            if start_frame_id <= frame_id < stop_frame_id:
                                img, h_ori, w_ori, _, orientation = image.loadImage(str(chunk_image_paths[frame_id - first_frame_id][0]), True)
                                x1, y1, x2, y2 = bboxUtils.box_to_display(box, source_info["PAR"])
                                img_buf = oiio.ImageBuf(img)
                                img_buf = oiio.ImageBufAlgo.crop(img_buf, roi=oiio.ROI(x1, x2, y1, y2))
                                img_crop = img_buf.get_pixels(format=oiio.FLOAT)
                                method, frame = self._resize_image(img_crop, chunk.node.inferenceSize.value)
                                resized_h, resized_w = frame.shape[:2]
                                mask_path = str(chunk_image_paths[frame_id - first_frame_id][1])
                                mask_path = mask_path.replace("%PROMPT%", text_prompt)
                                color_mask = True
                                if not os.path.exists(mask_path):
                                    mask_path = mask_path.replace("_merged_", "_fwd_")
                                    if not os.path.exists(mask_path):
                                        mask_path = mask_path.replace(f"colorMask_{text_prompt}_fwd_", "")
                                        color_mask = False
                                mask, _, _, _, _ = image.loadImage(mask_path, True, True, False)
                                img_buf = oiio.ImageBuf(mask)
                                if color_mask:
                                    mask_uint8 = np.rint(np.clip(mask * 255, 0, 255)).astype(np.uint8)
                                    color_index = 0 if obj_id=="" else int(obj_id)
                                    color_palette.generate_palette(color_index + 1)
                                    tgt = color_palette.at(color_index)
                                    mask_id = np.zeros_like(img, dtype=np.float32)
                                    mask_id[(mask_uint8 == tgt).all(axis = -1)] = [1.0, 1.0, 1.0]
                                    img_buf = oiio.ImageBuf(mask_id)

                                img_buf = oiio.ImageBufAlgo.crop(img_buf, roi=oiio.ROI(x1, x2, y1, y2))
                                img_crop = img_buf.get_pixels(format=oiio.FLOAT)
                                if method == "resize":
                                    mask = cv2.resize(img_crop, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
                                else:
                                    mask = self._padx8_image(img_crop)
                                cond_frames.append(frame)
                                mask_frames.append(mask)
                        nb_frames, shape = self._check_lists_compatibility(cond_frames, mask_frames)
                        logger.info(f"slice_idx = {slice_idx} ; {nb_frames} frames ; shape = {shape} ; method = {method}")

                        try:
                            with torch.amp.autocast('cuda', enabled=False):
                                output_frames = pipeline.run(cond_frames=cond_frames, mask_frames=mask_frames, seed=42)
                        except Exception as ex:
                            logger.error(f"Error in VideoMatting inference at slice {slice_idx}: {ex}")
                            raise

                        if slice_idx == 0:
                            mix_frames = output_frames[0:overlap]
                        else:
                            mix_frames = []
                            for i in range(overlap):
                                new_weight = (i + 1) / (overlap + 1)
                                blended_frame = (1.0 - new_weight) * previous_frames[i] + new_weight * output_frames[i].copy()
                                mix_frames.append(blended_frame)

                        if len(output_frames) >= overlap:
                            previous_frames = copy.deepcopy(output_frames[-overlap:])

                        if slice_idx > 0:
                            start_frame_id -= overlap
                        if slice_idx < len(time_slices) - 1:
                            stop_frame_id -= overlap

                        for frame_id, box in sorted(frame_chunk.boxes.items()):
                            if frame_id >= start_frame_id and frame_id < stop_frame_id:
                                frame_idx = frame_id - start_frame_id
                                if frame_idx < batch_size - overlap or slice_idx == len(time_slices) - 1:
                                    x1, y1, x2, y2 = bboxUtils.box_to_display(box, source_info["PAR"])
                                    box_w = x2 - x1
                                    box_h = y2 - y1
                                    output_frame = mix_frames[frame_idx] if frame_idx < overlap else output_frames[frame_idx].copy()
                                    alpha = self._restore_image_size(output_frame, (box_w, box_h), method)
                                    full_alpha[frame_id][y1:y2, x1:x2, :] += alpha

            for key, frame_chunks in bboxes_metadata.items():
                if "_" in key:
                    text_prompt, obj_id = key.rsplit('_', 1)
                else:
                    text_prompt, obj_id = key, ""
                for frame_chunk in frame_chunks:
                    for frame_idx, box in sorted(frame_chunk.boxes.items()):
                        if text_prompt not in metadata_boxes[frame_idx]:
                            metadata_boxes[frame_idx][text_prompt] = {}
                        x1, y1, x2, y2 = box
                        bbox_str = str(x1) + ";" + str(y1)+ ";" + str(x2)+ ";" + str(y2)
                        metadata_boxes[frame_idx][text_prompt][text_prompt + "_" + str(obj_id)] = bbox_str

            for frame_id, image_path in enumerate(chunk_image_paths):
                opt_write = avimg.ImageWriteOptions()
                opt_write.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
                if Path(image_path[4]).suffix.lower() == ".exr":
                    opt_write.exrCompressionMethod(avimg.EImageExrCompression_stringToEnum("DWAA"))
                    opt_write.exrCompressionLevel(300)

                frame_metadata_deep_model = dict(metadata_deep_model_base)
                for _, bboxes in metadata_boxes[first_frame_id + frame_id].items():
                    for k, box in bboxes.items():
                        frame_metadata_deep_model["Meshroom:mrSegmentation:" + k] = box
                alpha = np.clip(full_alpha[image_path[2]], 0, 1)
                image.writeImage(image_path[4], alpha, source_info["h_ori"], source_info["w_ori"], source_info["orientation"],
                                source_info["PAR"], frame_metadata_deep_model, opt_write)

        finally:
            torch.cuda.empty_cache()
