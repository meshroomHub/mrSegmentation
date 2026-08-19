__version__ = "2.1"

import copy
import logging
import os
from pathlib import Path
import re

from pyalicevision import parallelization as avpar
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

logger = logging.getLogger("VideoSegmentationSam3Text")

class VideoSegmentationSam3Text(desc.Node):
    """
Based on the Segment Anything video predictor model 3, the node generates a binary mask, a colored mask and an exr
cryptomatte from a text prompt.
"""
    size = avpar.DynamicViewsSize("input")
    gpu = lambda node: desc.Level.EXTREME if node.useOnlyHighPowerGpu.value else desc.Level.INTENSIVE

    text_prompts = []
    image_paths = []

    category = "Segmentation"

    inputs = [
        desc.File(
            name="input",
            description="SfMData file.",
            value="",
        ),
        desc.StringParam(
            name="prompt",
            description="What to segment, one item per line.",
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
            name="combineFwdAndBwdSeg",
            label="Combine Forward and Backward Segmentation",
            description="Launch segmentation in both forward and backward directions and combine masks.",
            value=False,
        ),
        desc.BoolParam(
            name="timeSlicing",
            description="Enable time slicing by adding text prompt every N frames and Propagating the masks over N frames.\n"
                        "Propagation is forward only by default, or both forward and backward when 'Combine Forward \n"
                        "and Backward Segmentation' is enabled.",
            value=False,
        ),
        desc.IntParam(
            name="sliceSize",
            description="Number of frames on which the mask is propagated.",
            value=16,
            enabled=lambda node: node.timeSlicing.value,
        ),
        desc.BoolParam(
            name="enableBonding",
            label="Enable Masks Bonding",
            description="Enable bonding where instances overlap.",
            value=True,
        ),
        desc.IntParam(
            name="bondingKernelSize",
            label="Bonding Kernel Size",
            description="Kernel size for morphological processing applied for masks bonding.",
            value=11,
            range=(1, 255, 2),
            enabled=lambda node: node.enableBonding.value,
        ),
        desc.BoolParam(
            name="maskInvert",
            label="Invert Masks",
            description="Invert mask values.\n"
                        "If selected, the pixels corresponding to the mask will be set to 0 instead of 255.",
            value=False,
        ),
        desc.BoolParam(
            name="outputCryptomatte",
            description="Generate exr images containing cryptomatte to encode the segmentation results.",
            value=False,
        ),
        desc.BoolParam(
            name="outputColorMasks",
            description="Generate colored masks where colors are linked with object Ids.",
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
            description="Generated segmentation masks.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + "." + attr.node.extensionOut.value,
        ),
        desc.File(
            name="colorMasksFwd",
            label="Colored Masks Forward",
            description="Colored segmentation masks resulting from forward propagation.\n"
                        "Colors correspond to instance indexes.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputColorMasks.value,
        ),
        desc.File(
            name="colorMasksBwd",
            label="Colored Masks Backward",
            description="Colored segmentation masks resulting from backward propagation.\n"
                        "Colors correspond to instance indexes.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputColorMasks.value and node.combineFwdAndBwdSeg.value,
        ),
        desc.File(
            name="colorMasksMerged",
            label="Colored Masks Merged",
            description="Colored segmentation masks resulting from merging forward and backward propagation.\n"
                        "Colors correspond to instance indexes.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputColorMasks.value and node.combineFwdAndBwdSeg.value,
        ),
        desc.File(
            name="cryptomatte",
            label="Cryptomatte",
            description="Cryptomatte embedded in EXR images resulting from merged result if available, else from forward propagation only.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputCryptomatte.value,
        ),
    ]

    def _build_output_path(self, node, frame_id, prefix, extension):
        """ Constructs an absolute output filepath based on the preferred naming methods. """
        if node.keepFilename.value:
            path = str(Path(self.image_paths[frame_id][0]).stem)
        else:
            path = str(self.image_paths[frame_id][1])
        return os.path.join(node.output.value, prefix + path + extension)

    def _update_global_ids(self, results, prev_overlap, next_id, color_palette):
        """ Assigns global tracking IDs and expands the color palette. """
        from segmentationRDS import sam3Utils

        updated_res, next_prev_overlap, updated_id = sam3Utils.assign_global_ids(
            results,
            prev_overlap,
            next_id,
            similarity_threshold=0.4,
            use_mask=True,
        )
        color_palette.generate_palette(updated_id + 1)
        return updated_res, next_prev_overlap, updated_id

    def _slice_track(self, track_data, start, end):
        """ Helper to slice tracking dictionaries between frame bounds. """
        return {k: v for k, v in track_data.items() if start <= k <= end}

    def _get_tracking_config(self, node, frame_number):
        """ Determines directional limits, frame step intervals and slicing logic. """
        frame_idx_to_text_prompt = [0]
        max_frame_num_to_track = None
        track_dir = "forward"

        # Construct frame sequence intervals if time slicing is active
        if node.timeSlicing.value:
            max_frame_num_to_track = node.sliceSize.value
            curr_frame_to_text_prompt = 0
            while curr_frame_to_text_prompt + node.sliceSize.value < frame_number:
                curr_frame_to_text_prompt += node.sliceSize.value
                frame_idx_to_text_prompt.append(curr_frame_to_text_prompt)

        if node.combineFwdAndBwdSeg.value:
            track_dir = "both"
            if frame_idx_to_text_prompt[-1] < frame_number - 1:
                frame_idx_to_text_prompt.append(frame_number - 1)

        return frame_idx_to_text_prompt, max_frame_num_to_track, track_dir

    def _load_source_images(self):
        """ Loads input images, generates baseline empty masks, and indexes camera dimensions. """
        from PIL import Image
        from segmentationRDS import image
        import numpy as np

        pil_images = []
        mask_images = []
        source_info = None

        for idx, path_data in enumerate(self.image_paths):
            img, h_ori, w_ori, par, orientation = image.loadImage(str(path_data[0]), True)
            pil_images.append(Image.fromarray((255.0 * img).astype("uint8")))

            # Store source dimensions from the first image (assumed uniform)
            if idx == 0:
                source_info = {
                    "h_ori": h_ori,
                    "w_ori": w_ori,
                    "PAR": par,
                    "orientation": orientation,
                    "shape": img.shape,
                    "dtype": img.dtype
                }
            mask_images.append(np.zeros((*img.shape[:2], 1), dtype=img.dtype))

        return pil_images, mask_images, source_info

    def _export_direction_masks(
        self,
        node,
        frame_range,
        direction_name,
        direction_results,
        text_prompt,
        color_palette,
        source_info,
        state,
        metadata_deep_model
    ):
        """ Helper to process results, draw color masks, generate cryptomatte, and save outputs. """
        from pyalicevision import image as avimg
        from segmentationRDS import image, sam3Utils
        import numpy as np

        prefix_map = {
            "forward": "fwd",
            "backward": "bwd",
            "merged": "merged"
        }
        dir_prefix = prefix_map[direction_name]

        ext_map = {
            "forward": ".exr",
            "backward": ".png",
            "merged": ".exr"
        }
        color_mask_ext = ext_map[direction_name]

        first_frame_id = source_info["first_frame_id"]

        mask_images = state["mask_images"]
        boxes = state["boxes"]
        metadata_boxes = state["metadata_boxes"]

        is_definitive = (
            (node.combineFwdAndBwdSeg.value and direction_name == "merged") or
            (not node.combineFwdAndBwdSeg.value and direction_name == "forward")
        )

        crypto_name = "cryptoObject" if text_prompt == "" else text_prompt.replace(" ", "_")

        for frame_id in frame_range:
            color_mask_image = np.zeros(source_info["shape"], dtype=source_info["dtype"])

            if (first_frame_id + frame_id) not in boxes[text_prompt][direction_name]:
                boxes[text_prompt][direction_name][first_frame_id + frame_id] = {}

            output_crypto = node.outputCryptomatte.value and is_definitive
            if output_crypto:
                crypto_id = np.zeros((source_info["h_ori"], source_info["w_ori"]), dtype=np.float32)
                crypto_cov = np.zeros((source_info["h_ori"], source_info["w_ori"]), dtype=np.float32)
                manifest = {}

            # Get detections for current frame (if any)
            frame_detections = direction_results.get(frame_id, {})

            masks = []
            colors = []
            for key, mask_box_prob in frame_detections.items():
                mask = mask_box_prob["mask"]
                masks.append(mask.squeeze())

                # Draw visual color mask
                color = color_palette.at(int(key)) if color_palette.at(int(key)) is not None else [255, 255, 255]
                color_normalized = [x / 255.0 for x in color]
                colors.append(color_normalized)
                color_mask_image[mask] = color_normalized

                # Generate IDs and hash structures for cryptomatte EXRs
                if output_crypto:
                    obj_name = f"{crypto_name}_{int(key)}"
                    f32_hash, hex_val, _ = image.hash_name(obj_name)
                    manifest[obj_name] = hex_val
                    crypto_id[mask] = f32_hash
                    crypto_cov[mask] = 1.0

                # Process Bounding Box mapping
                bbox = sam3Utils.xywhNorm2xyxy(mask_box_prob["box_xywh"], source_info["w_ori"], source_info["h_ori"])
                boxes[text_prompt][direction_name][first_frame_id + frame_id][key] = bbox

                # Write the bounding boxes into the frame's metadata
                x1, y1, x2, y2 = bbox
                bbox_str = f"{x1};{y1};{x2};{y2}"
                metadata_boxes[frame_id][text_prompt][direction_name][f"{dir_prefix}_{text_prompt}_{key}"] = bbox_str

            if masks:
                masks_stack = np.stack(masks, axis=0)
                mask_global = np.expand_dims(np.sum(masks_stack, axis=0), axis=-1).astype(np.float32)

                if len(masks) > 1 and node.enableBonding.value:
                    ks = node.bondingKernelSize.value
                    bonded_bin, bonded_color = sam3Utils.bond_masks(
                        masks=masks,
                        colors=colors,
                        dilate_kernel_size=ks,
                        conflict_kernel_size=ks,
                        close_kernel_size=ks
                    )
                    mask_global = np.expand_dims(bonded_bin, axis=-1)
                    color_mask_image = bonded_color

                if is_definitive:
                    mask_images[frame_id] = np.maximum(mask_images[frame_id], mask_global, out=mask_images[frame_id])

            # Save color mask image
            if node.outputColorMasks.value:
                prefix = f"colorMask_{text_prompt}_{dir_prefix}_"
                output_file_color_mask = self._build_output_path(node, frame_id, prefix, color_mask_ext)
                opt_write = avimg.ImageWriteOptions()
                opt_write.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
                image.writeImage(
                    output_file_color_mask,
                    color_mask_image,
                    source_info["h_ori"],
                    source_info["w_ori"],
                    source_info["orientation"],
                    source_info["PAR"],
                    metadata_deep_model,
                    opt_write
                )

            # Save Cryptomatte Multichannel EXR
            if output_crypto:
                prefix = f"cryptomatte_{text_prompt}_{dir_prefix}_"
                cryptomatte_path = self._build_output_path(node, frame_id, prefix, ".exr")
                image.writeCryptomatte(
                    cryptomatte_path,
                    crypto_name,
                    source_info["w_ori"],
                    source_info["h_ori"],
                    manifest,
                    crypto_id,
                    crypto_cov,
                    color_mask_image
                )

    def _update_tracking_at_step(
        self,
        n,
        frame_idx,
        frame_idx_to_text_prompt,
        outputs_per_frame,
        track_states,
        color_palette,
        combine_fwd_bwd
    ):
        """ Processes temporal tracking slices, resolves global IDs, and manages historical memory. """
        from segmentationRDS import sam3Utils

        # Call prepareMasksForVisualization once per raw outputs dictionary to prevent repeating mutating calls
        prepared_track = sam3Utils.prepareMasksForVisualization(outputs_per_frame[frame_idx])

        # n == 0: Initialization block
        if n == 0:
            # Deep copy to prevent in-place modifications from leaking between lists
            fwd_only = copy.deepcopy(prepared_track)
            bwd_only = copy.deepcopy(prepared_track)
            fwd_bwd = None

            # Properly initialize the Forward global ID tracking at n = 0
            first_frame = frame_idx
            last_frame = frame_idx_to_text_prompt[1] if len(frame_idx_to_text_prompt) > 1 else max(fwd_only.keys())
            fwd = self._slice_track(fwd_only, first_frame, last_frame)
            fwd_only, track_states["fwd"]["prev_overlap"], track_states["fwd"]["next_id"] = self._update_global_ids(
                fwd, track_states["fwd"]["prev_overlap"], track_states["fwd"]["next_id"], color_palette
            )
            logger.info(f"next_global_id_fwd = {track_states['fwd']['next_id']}")

            # Initialize backward tracking
            bwd_frame0_only = {frame_idx: bwd_only[frame_idx]}
            _, track_states["bwd"]["prev_overlap"], track_states["bwd"]["next_id"] = self._update_global_ids(
                bwd_frame0_only, track_states["bwd"]["prev_overlap"], track_states["bwd"]["next_id"], color_palette
            )
            logger.info(f"next_global_id_bwd = {track_states['bwd']['next_id']}")

            # Initialize merged tracking if enabled
            if combine_fwd_bwd:
                merged_frame0_only = {frame_idx: fwd_only[frame_idx]}
                fwd_bwd, track_states["merged"]["prev_overlap"], track_states["merged"]["next_id"] = self._update_global_ids(
                    merged_frame0_only, track_states["merged"]["prev_overlap"],
                    track_states["merged"]["next_id"], color_palette
                )
                logger.info(f"next_global_id_merged = {track_states['merged']['next_id']}")

            return fwd_only, bwd_only, fwd_bwd

        # n > 0: Propagation & Merging block
        track_fwd = sam3Utils.prepareMasksForVisualization(outputs_per_frame[frame_idx])
        first_frame = frame_idx

        # Keep the remaining forward frames on the last segment instead of truncating to frame_idx
        if n == len(frame_idx_to_text_prompt) - 1:
            last_frame = max(track_fwd.keys())
        else:
            last_frame = frame_idx_to_text_prompt[n + 1]
        fwd = self._slice_track(track_fwd, first_frame, last_frame)

        fwd_only, track_states["fwd"]["prev_overlap"], track_states["fwd"]["next_id"] = self._update_global_ids(
            fwd, track_states["fwd"]["prev_overlap"], track_states["fwd"]["next_id"], color_palette
        )
        logger.info(f"next_global_id_fwd = {track_states['fwd']['next_id']}")

        bwd_only = None
        fwd_bwd = None

        if combine_fwd_bwd:
            track_bwd = sam3Utils.prepareMasksForVisualization(outputs_per_frame[frame_idx])
            first_frame_bwd = frame_idx_to_text_prompt[n - 1]
            last_frame_bwd = frame_idx
            bwd = self._slice_track(track_bwd, first_frame_bwd, last_frame_bwd)

            bwd_only, track_states["bwd"]["prev_overlap"], track_states["bwd"]["next_id"] = self._update_global_ids(
                bwd, track_states["bwd"]["prev_overlap"], track_states["bwd"]["next_id"], color_palette
            )
            logger.info(f"next_global_id_bwd = {track_states['bwd']['next_id']}")

            # Fresh, completely isolated slice calculations for merge operations
            prev_prepared_track = sam3Utils.prepareMasksForVisualization(outputs_per_frame[frame_idx_to_text_prompt[n - 1]])
            track_fwd_for_merge = copy.deepcopy(prev_prepared_track)
            track_bwd_for_merge = copy.deepcopy(prepared_track)
            fwd_for_merge = self._slice_track(track_fwd_for_merge, first_frame_bwd, last_frame_bwd)
            bwd_for_merge = self._slice_track(track_bwd_for_merge, first_frame_bwd, last_frame_bwd)

            fwd_bwd, track_states["merged"]["prev_overlap"], track_states["merged"]["next_id"] = sam3Utils.merge_tracks(
                fwd_results=fwd_for_merge,
                bwd_results=bwd_for_merge,
                prev_overlap_detections=track_states["merged"]["prev_overlap"],
                next_global_id=track_states["merged"]["next_id"],
                similarity_threshold_merge=0.3,
                similarity_threshold_track=0.4,
                use_mask=True,
            )
            color_palette.generate_palette(track_states["merged"]["next_id"] + 1)
            logger.info(f"next_global_id_merged = {track_states['merged']['next_id']}")

        # Clear historical frames from memory
        del outputs_per_frame[frame_idx_to_text_prompt[n - 1]]

        return fwd_only, bwd_only, fwd_bwd

    def resolve_paths(self, node):
        """ Constructs system template paths mapping to the node output parameters. """
        if node.prompt.value == "":
            raise ValueError("Text prompt is empty.")

        input_path = node.input.value
        image_paths = get_image_paths_list(input_path)
        if len(image_paths) == 0:
            raise FileNotFoundError(f"No image files found in {input_path}.")
        self.image_paths = image_paths

        # Parse and sanitize multi-line prompt lists
        self.text_prompts = re.split(r'[\n]+', node.prompt.value)
        self.text_prompts = [str(text_prompt) for text_prompt in self.text_prompts if text_prompt]

        src_filename = "<FILESTEM>" if node.keepFilename.value else "<VIEW_ID>"

        color_mask_prefix = node.output.value + "/colorMask_" + self.text_prompts[0]
        cryptomatte_prefix = node.output.value + "/cryptomatte_" + self.text_prompts[0]

        node.colorMasksFwd.value = color_mask_prefix + "_fwd_" + src_filename + ".exr"
        node.colorMasksBwd.value = color_mask_prefix + "_bwd_" + src_filename + ".png"
        node.colorMasksMerged.value = color_mask_prefix + "_merged_" + src_filename + ".exr"
        node.cryptomatte.value = cryptomatte_prefix + ("_merged_" if node.combineFwdAndBwdSeg.value else "_fwd_") + src_filename + ".exr"

    def processChunk(self, chunk):
        from segmentationRDS import image, sam3Utils
        from sam3.model_builder import build_sam3_video_predictor
        import torch
        from pyalicevision import image as avimg
        import json

        try:
            self.resolve_paths(chunk.node)

            logger.setLevel(chunk.node.verboseLevel.value.upper())

            if not chunk.node.input:
                logger.warning("Nothing to segment")
                return
            if not chunk.node.output.value:
                return

            logger.info(f"Chunk range from {chunk.range.start} to {chunk.range.last}")

            if not os.path.exists(chunk.node.output.value):
                os.mkdir(chunk.node.output.value)

            gpus_to_use = [torch.cuda.current_device()]
            video_predictor = build_sam3_video_predictor(checkpoint_path=chunk.node.segmentationModelPath.evalValue,
                                                         gpus_to_use=gpus_to_use)

            metadata_deep_model = {}
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelName"] = "SegmentAnything"
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelVersion"] = "sam3-Video-TextPrompt"

            color_palette = image.paletteGenerator()
            first_frame_id = self.image_paths[0][2]
            frame_number = len(self.image_paths)

            frame_idx_to_text_prompt, max_frame_num_to_track, track_dir = self._get_tracking_config(chunk.node, frame_number)
            logger.info(f"frame_idx_to_text_prompt: {frame_idx_to_text_prompt}; direction = {track_dir}")

            pil_images, mask_images, source_info = self._load_source_images()
            source_info["first_frame_id"] = first_frame_id

            # Start tracking session
            response = video_predictor.handle_request(
                request={"type": "start_session", "resource_path": pil_images}
            )
            session_id = response["session_id"]

            boxes = {}
            metadata_boxes = {frame_id: {} for frame_id in range(frame_number)}

            tracking_state = {
                "mask_images": mask_images,
                "boxes": boxes,
                "metadata_boxes": metadata_boxes
            }

            # Run temporal tracking queries per text prompt configuration
            for text_prompt in self.text_prompts:
                logger.info(f"Processing prompt: {text_prompt}")

                boxes[text_prompt] = {"forward": {}, "backward": {}, "merged": {}}
                metadata_deep_model["Meshroom:mrSegmentation:Prompt"] = text_prompt

                for frame_id in range(frame_number):
                    metadata_boxes[frame_id][text_prompt] = {"forward": {}, "backward": {}, "merged": {}}

                video_predictor.handle_request(request={"type": "reset_session", "session_id": session_id})

                outputs_per_frame = {}

                track_states = {
                    "fwd": {"prev_overlap": {}, "next_id": 0},
                    "bwd": {"prev_overlap": {}, "next_id": 0},
                    "merged": {"prev_overlap": {}, "next_id": 0}
                }

                for n, frame_idx in enumerate(frame_idx_to_text_prompt):
                    abs_frame = int(first_frame_id) + frame_idx
                    logger.info(f"Text prompt at relative frame {frame_idx} (absolute frame {abs_frame})")

                    video_predictor.handle_request(
                        request={
                            "type": "add_prompt",
                            "session_id": session_id,
                            "frame_index": frame_idx,
                            "text": text_prompt
                        }
                    )
                    outputs_per_frame[frame_idx] = sam3Utils.propagateInVideo(video_predictor, session_id,
                                                                              frame_idx, max_frame_num_to_track,
                                                                              track_dir)

                    fwd_only, bwd_only, fwd_bwd = self._update_tracking_at_step(
                        n=n,
                        frame_idx=frame_idx,
                        frame_idx_to_text_prompt=frame_idx_to_text_prompt,
                        outputs_per_frame=outputs_per_frame,
                        track_states=track_states,
                        color_palette=color_palette,
                        combine_fwd_bwd=chunk.node.combineFwdAndBwdSeg.value
                    )

                    # write Fwd from frame_idx to frame_idx_to_text_prompt[n + 1]
                    last_frame_idx_fwd = frame_number
                    if n < len(frame_idx_to_text_prompt) - 1:
                        last_frame_idx_fwd = frame_idx_to_text_prompt[n + 1]
                    logger.debug(f"Exporting forward boxes from frame index {frame_idx} to {last_frame_idx_fwd - 1}")

                    self._export_direction_masks(
                        node=chunk.node,
                        frame_range=range(frame_idx, last_frame_idx_fwd),
                        direction_name="forward",
                        direction_results=fwd_only,
                        text_prompt=text_prompt,
                        color_palette=color_palette,
                        source_info=source_info,
                        state=tracking_state,
                        metadata_deep_model=metadata_deep_model
                    )

                    if chunk.node.combineFwdAndBwdSeg.value:
                        # write Bwd from frame_idx_to_text_prompt[n - 1] to frame_idx
                        first_frame_idx_bwd = frame_idx_to_text_prompt[n - 1] + 1 if n > 0 else frame_idx

                        self._export_direction_masks(
                            node=chunk.node,
                            frame_range=range(first_frame_idx_bwd, frame_idx + 1),
                            direction_name="backward",
                            direction_results=bwd_only,
                            text_prompt=text_prompt,
                            color_palette=color_palette,
                            source_info=source_info,
                            state=tracking_state,
                            metadata_deep_model=metadata_deep_model
                        )

                        if n > 0:
                            self._export_direction_masks(
                                node=chunk.node,
                                frame_range=range(first_frame_idx_bwd - 1, frame_idx + 1),
                                direction_name="merged",
                                direction_results=fwd_bwd,
                                text_prompt=text_prompt,
                                color_palette=color_palette,
                                source_info=source_info,
                                state=tracking_state,
                                metadata_deep_model=metadata_deep_model
                            )

            prompts = [text_prompt.strip() for text_prompt in self.text_prompts if text_prompt.strip()]
            metadata_deep_model["Meshroom:mrSegmentation:Prompt"] = ";".join(prompts)

            logger.info("Writing definitive binary masks to disk...")
            for frame_id in range(frame_number):
                if chunk.node.maskInvert.value:
                    mask = (mask_images[frame_id][:, :, 0:1] == 0).astype('float32')
                else:
                    mask = (mask_images[frame_id][:, :, 0:1] > 0).astype('float32')
                logger.info(f"frame_id: {frame_id} - {self.image_paths[frame_id][0]}")

                output_file_mask = self._build_output_path(chunk.node, frame_id, "", "." + chunk.node.extensionOut.value)
                opt_write = avimg.ImageWriteOptions()
                opt_write.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)

                if Path(output_file_mask).suffix.lower() == ".exr":
                    opt_write.exrCompressionMethod(avimg.EImageExrCompression_stringToEnum("DWAA"))
                    opt_write.exrCompressionLevel(300)

                frame_metadata_deep_model = dict(metadata_deep_model)
                for prompt in self.text_prompts:
                    for direction in ["forward", "backward", "merged"]:
                        for k, box in metadata_boxes[frame_id][prompt][direction].items():
                            frame_metadata_deep_model["Meshroom:mrSegmentation:" + k] = box

                image.writeImage(output_file_mask, mask, source_info["h_ori"],
                                 source_info["w_ori"], source_info["orientation"],
                                 source_info["PAR"], frame_metadata_deep_model, opt_write)

            json_filename = chunk.node.output.value + "/bboxes.json"
            logger.info(f"Writing bounding boxes metadata to {json_filename}")
            with open(json_filename, "w", encoding="utf_8") as file:
                json.dump(boxes, file, indent=4, ensure_ascii=False)

            video_predictor.handle_request(request={"type": "close_session", "session_id": session_id})

        finally:
            torch.cuda.empty_cache()


def get_image_paths_list(input_path):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO

    image_paths = []

    if Path(input_path).suffix.lower() in [".sfm", ".abc"]:
        if Path(input_path).exists():
            data = sfmData.SfMData()
            if sfmDataIO.load(data, input_path, sfmDataIO.ALL):
                views = data.getViews()
                for view_id, view in views.items():
                    image_paths.append((Path(view.getImage().getImagePath()), str(view_id), view.getFrameId()))

            image_paths.sort(key=lambda x: x[0])
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path (folder or sfmData file).")
    return image_paths
