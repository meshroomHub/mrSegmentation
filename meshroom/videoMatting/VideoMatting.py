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
                    output_cryptomatte_path = os.path.join(output_dir, "cryptomatte_" + filename + "." + output_ext)
                else:
                    if mask_path:
                        input_file_mask = os.path.join(mask_path, str(view_id) + "." + mask_ext)
                    output_file_matte = os.path.join(output_dir, str(view_id) + "." + output_ext)
                    output_cryptomatte_path = os.path.join(output_dir, "cryptomatte_" + str(view_id) + "." + output_ext)
                paths.append((input_file, input_file_mask, frame_id, str(view_id), output_file_matte,
                              output_cryptomatte_path, img_width, img_height, par))
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

    def _read_exr_first_channel(self, path: str):
        import OpenImageIO as oiio
        import numpy as np

        inp = oiio.ImageInput.open(str(path))
        if not inp:
            raise RuntimeError(f"Cannot open EXR: {path}")
        try:
            arr = inp.read_image(format=oiio.FLOAT)  # float32, shape (h,w,nch) or (h,w)
            if arr is None:
                raise RuntimeError(f"Cannot read EXR: {path}")
        finally:
            inp.close()

        arr = np.asarray(arr)
        if arr.ndim == 2:
            return arr.astype(np.float32, copy=False)
        return arr[..., 0].astype(np.float32, copy=False)

    def _update_top4_inplace(self, ids4, cov4, roi_id, roi_cov, eps: float = 1e-8):
        """
        ids4, cov4: (h,w,4) views into global buffers
        roi_id: float32 cryptomatte id
        roi_cov: (h,w) float32 mask values (non-binary ok)
        Keeps top-4 by coverage per pixel (in descending order).
        """
        import numpy as np

        m = roi_cov > eps
        if not np.any(m):
            return

        c = roi_cov[m]

        c0 = cov4[..., 0][m]; i0 = ids4[..., 0][m]
        c1 = cov4[..., 1][m]; i1 = ids4[..., 1][m]
        c2 = cov4[..., 2][m]; i2 = ids4[..., 2][m]
        c3 = cov4[..., 3][m]; i3 = ids4[..., 3][m]

        gt0 = c > c0
        if np.any(gt0):
            c3[gt0], i3[gt0] = c2[gt0], i2[gt0]
            c2[gt0], i2[gt0] = c1[gt0], i1[gt0]
            c1[gt0], i1[gt0] = c0[gt0], i0[gt0]
            c0[gt0], i0[gt0] = c[gt0],  roi_id

        mid1 = (~gt0) & (c > c1)
        if np.any(mid1):
            c3[mid1], i3[mid1] = c2[mid1], i2[mid1]
            c2[mid1], i2[mid1] = c1[mid1], i1[mid1]
            c1[mid1], i1[mid1] = c[mid1],  roi_id

        mid2 = (~gt0) & (~mid1) & (c > c2)
        if np.any(mid2):
            c3[mid2], i3[mid2] = c2[mid2], i2[mid2]
            c2[mid2], i2[mid2] = c[mid2],  roi_id

        low = (~gt0) & (~mid1) & (~mid2) & (c > c3)
        if np.any(low):
            c3[low], i3[low] = c[low], roi_id

        cov4[..., 0][m] = c0; ids4[..., 0][m] = i0
        cov4[..., 1][m] = c1; ids4[..., 1][m] = i1
        cov4[..., 2][m] = c2; ids4[..., 2][m] = i2
        cov4[..., 3][m] = c3; ids4[..., 3][m] = i3

    def _write_cryptomatte_top4_nuke(self, filepath: str,
                                     crypto_layer_name: str,
                                     W: int, H: int,
                                     manifest: dict,
                                     ids4,
                                     cov4,
                                     preview_rgb = None):
        """
        Writes EXR with:
        - Optional preview RGBA in base channels (R,G,B,A)
        - 2 cryptomatte layers:
            cryptoName00 rgba = (id0,cov0,id1,cov1)
            cryptoName01 rgba = (id2,cov2,id3,cov3)
        """
        import OpenImageIO as oiio
        import numpy as np
        import json
        from segmentationRDS import image

        has_preview = preview_rgb is not None
        nchan = (4 if has_preview else 0) + 8

        spec = oiio.ImageSpec(W, H, nchan, oiio.FLOAT)

        ch = []
        if has_preview:
            ch += ["R", "G", "B", "A"]
        ch += [
            f"{crypto_layer_name}00.red", f"{crypto_layer_name}00.green", f"{crypto_layer_name}00.blue",  f"{crypto_layer_name}00.alpha",
            f"{crypto_layer_name}01.red", f"{crypto_layer_name}01.green", f"{crypto_layer_name}01.blue",  f"{crypto_layer_name}01.alpha",
        ]
        spec.channelnames = ch

        # Cryptomatte metadata
        _, _, h32 = image.hash_name(crypto_layer_name)
        crypto_key = f"{h32 & 0xFFFFFFFF:08x}"[:7]
        spec.attribute(f"cryptomatte/{crypto_key}/name", crypto_layer_name)
        spec.attribute(f"cryptomatte/{crypto_key}/manifest", json.dumps(manifest, separators=(",", ":")))
        spec.attribute(f"cryptomatte/{crypto_key}/hash", "MurmurHash3_32")
        spec.attribute(f"cryptomatte/{crypto_key}/conversion", "uint32_to_float32")
        spec.attribute(f"cryptomatte/{crypto_key}/version", "1.0")

        parts = []

        if has_preview:
            preview_rgb = np.asarray(preview_rgb, dtype=np.float32)
            if preview_rgb.shape != (H, W, 3):
                raise ValueError(f"preview_rgb must be ({H},{W},3), got {preview_rgb.shape}")
            alpha = np.ones((H, W, 1), dtype=np.float32)
            parts.append(np.concatenate([preview_rgb, alpha], axis=2))

        # Pack top-4 into 2 RGBA layers (Nuke convention)
        id0, id1, id2, id3 = (ids4[..., 0], ids4[..., 1], ids4[..., 2], ids4[..., 3])
        c0,  c1,  c2,  c3  = (cov4[..., 0], cov4[..., 1], cov4[..., 2], cov4[..., 3])

        crypto00 = np.dstack((id0, c0, id1, c1))
        crypto01 = np.dstack((id2, c2, id3, c3))
        parts.append(crypto00)
        parts.append(crypto01)

        data = np.dstack(parts).astype(np.float32, copy=False)

        out = oiio.ImageOutput.create(str(filepath))
        if not out:
            raise RuntimeError(f"Cannot create ImageOutput for {filepath}")
        if not out.open(str(filepath), spec):
            err = out.geterror()
            out.close()
            raise RuntimeError(f"Cannot open {filepath}: {err}")
        ok = out.write_image(data)
        err = out.geterror()
        out.close()
        if not ok:
            raise RuntimeError(f"Write failed: {err}")
        
    def _build_cryptomatte_for_frame_top4(self, frame: int,
                                         mask_infos,
                                         out_path: str,
                                         crypto_layer_name: str,
                                         H: int, W: int,
                                         preview_rgb = None,
                                         eps: float = 1e-8,
                                         clamp01: bool = True,
                                         normalize_if_sum_gt_1: bool = True):
        """
        Builds top-4 cryptomatte buffers (H,W,4) by reading ROIs for the given frame.
        objectName_objectId is unique per frame => one ROI per object.
        """
        import numpy as np
        from segmentationRDS import image

        ids4 = np.zeros((H, W, 4), dtype=np.float32)
        cov4 = np.zeros((H, W, 4), dtype=np.float32)
        manifest: dict[str, str] = {}

        for mi in mask_infos:
            if mi["frame"] != frame:
                continue

            obj_name = mi["obj_name"]
            obj_id = mi["obj_id"]
            x1 = mi["x1"]
            y1 = mi["y1"]
            x2 = mi["x2"]
            y2 = mi["y2"]
            path = mi["path"]

            key = f"{obj_name}_{obj_id}"

            # bounds check (0-based, inclusive x2/y2)
            if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
                raise ValueError(f"ROI out of bounds: {path} ROI={(x1,x2,y1,y2)} frame={(W,H)}")

            # id + manifest
            f32_hash, hex_val, _ = image.hash_name(key)
            roi_id = np.float32(f32_hash)
            manifest[key] = hex_val

            roi = self._read_exr_first_channel(str(path))
            if roi.shape != (y2 - y1, x2 - x1):
                raise ValueError(f"Mask dims mismatch: {path} expected {(y2 - y1, x2 - x1)} got {roi.shape}")

            roi_cov = roi
            if clamp01:
                roi_cov = np.clip(roi_cov, 0.0, 1.0)

            # update only the ROI window
            ids_roi = ids4[y1:y2, x1:x2, :]
            cov_roi = cov4[y1:y2, x1:x2, :]

            self._update_top4_inplace(ids_roi, cov_roi, roi_id, roi_cov.astype(np.float32, copy=False), eps=eps)

            os.remove(path)

        if normalize_if_sum_gt_1:
            s = cov4.sum(axis=2, keepdims=True)
            den = np.maximum(s, 1.0)  # s<1 => den=1, donc cov4 inchangé
            cov4 = (cov4 / den).astype(np.float32, copy=False)

        self._write_cryptomatte_top4_nuke(
            out_path, crypto_layer_name, W, H,
            manifest=manifest,
            ids4=ids4, cov4=cov4,
            preview_rgb=preview_rgb
        )

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
            frame_w = chunk_image_paths[0][6]
            frame_h = chunk_image_paths[0][7]
            par = chunk_image_paths[0][8]
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
            masks_by_frame = {}
            img, h_ori, w_ori, p_a_r, orientation = image.loadImage(str(chunk_image_paths[0][0]), True)
            source_info = {"h_ori": h_ori, "w_ori": w_ori, "PAR": p_a_r, "orientation": orientation}
            for frame_id, image_path in enumerate(chunk_image_paths):
                full_alpha[image_path[2]] = np.zeros_like(img)
                masks_by_frame[int(image_path[2])] = []

            batch_size = chunk.node.batchSize.value
            overlap = chunk.node.overlap.value

            color_palette = image.paletteGenerator()

            for key, frame_chunks in bboxes.items():
                if "_" in key:
                    text_prompt, obj_id = key.rsplit('_', 1)
                else:
                    text_prompt, obj_id = key, "0"
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
                                    obj_name = text_prompt.replace(" ", "_")
                                    roi_path = os.path.join(chunk.node.output.value,
                                                            f"{str(frame_id)}%{obj_name}%{obj_id}%{str(x1)}%{str(y1)}%{str(x2)}%{str(y2)}.exr")
                                    image.write_exr_hxwx1_float_lossless(roi_path, alpha[:,:,0])
                                    masks_by_frame[frame_id].append({"frame": int(frame_id),
                                                                     "obj_name": obj_name,
                                                                     "obj_id": obj_id,
                                                                     "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                                                     "path": roi_path})

            for key, frame_chunks in bboxes_metadata.items():
                if "_" in key:
                    text_prompt, obj_id = key.rsplit('_', 1)
                else:
                    text_prompt, obj_id = key, "0"
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
                
                masks_infos = masks_by_frame.get(first_frame_id + frame_id, [])
                self._build_cryptomatte_for_frame_top4(first_frame_id + frame_id,
                                                       masks_infos,
                                                       image_path[5],
                                                       "cryptoObject",
                                                       source_info["h_ori"], source_info["w_ori"],
                                                       alpha)

        finally:
            torch.cuda.empty_cache()
