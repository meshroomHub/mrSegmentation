__version__ = "1.2"

import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
from pyalicevision import parallelization as avpar

import logging
logger = logging.getLogger("VideoSegmentationSam3Text")

class VideoSegmentationSam3Text(desc.Node):
    size = avpar.DynamicViewsSize("input")
    gpu = lambda node: desc.Level.EXTREME if node.useOnlyHighPowerGpu.value else desc.Level.INTENSIVE

    category = "Segmentation"
    documentation = """
Based on the Segment Anything video predictor model 3, the node generates a binary mask, a colored mask and an exr cryptomatte
from a text prompt.
"""

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
                        "Propagation is forward only by default, or both forward and backward when 'Combine Forward and Backward Segmentation'\n"
                        "is enabled.",
            value=False,
        ),
        desc.IntParam(
            name="sliceSize",
            description="Number of frames on which the mask is propagated.",
            value=16,
            enabled=lambda node: node.timeSlicing.value,
        ),
        desc.BoolParam(
            name="maskInvert",
            label="Invert Masks",
            description="Invert mask values. If selected, the pixels corresponding to the mask will be set to 0 instead of 255.",
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
            description="Colored segmentation masks resulting from forward propagation. Colors correspond to instance indexes.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputColorMasks.value,
        ),
        desc.File(
            name="colorMasksBwd",
            label="Colored Masks Backward",
            description="Colored segmentation masks resulting from backward propagation. Colors correspond to instance indexes.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputColorMasks.value and node.combineFwdAndBwdSeg.value,
        ),
        desc.File(
            name="colorMasksMerged",
            label="Colored Masks Merged",
            description="Colored segmentation masks resulting from merging forward and backward propagation. Colors correspond to instance indexes.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputColorMasks.value and node.combineFwdAndBwdSeg.value,
        ),
        desc.File(
            name="cryptomatteFwd",
            label="Cryptomatte Forward",
            description="Cryptomatte resulting from forward propagation embedded in EXR images.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputCryptomatte.value,
        ),
        desc.File(
            name="cryptomatteBwd",
            label="Cryptomatte Backward",
            description="Cryptomatte resulting from backward propagation embedded in EXR images.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputCryptomatte.value and node.combineFwdAndBwdSeg.value,
        ),
    ]

    def resolvePaths(self, node):
        import re

        if node.prompt.value == "":
            raise ValueError("Text prompt is empty.")

        input_path = node.input.value
        image_paths = get_image_paths_list(input_path)
        if len(image_paths) == 0:
            raise FileNotFoundError(f"No image files found in {input_path}.")
        self.image_paths = image_paths

        self.text_prompts = re.split(r'[\n]+', node.prompt.value)
        self.text_prompts = [str(text_prompt) for text_prompt in self.text_prompts if text_prompt]
        src_filename = "<FILESTEM>" if node.keepFilename.value else "<VIEW_ID>"
        color_mask_prefix = node.output.value + "/colorMask_" + self.text_prompts[0]
        cryptomatte_prefix = node.output.value + "/cryptomatte_" + self.text_prompts[0]
        node.colorMasksFwd.value = color_mask_prefix + "_fwd_" + src_filename + ".exr"
        node.colorMasksBwd.value = color_mask_prefix + "_bwd_" + src_filename + ".png"
        node.colorMasksMerged.value = color_mask_prefix + "_merged_" + src_filename + ".exr"
        node.cryptomatteFwd.value = cryptomatte_prefix + "_fwd_" + src_filename + ".png"
        node.cryptomatteBwd.value = cryptomatte_prefix + "_bwd_" + src_filename + ".png"


    def processChunk(self, chunk):
        from segmentationRDS import image, sam3Utils
        from sam3.model_builder import build_sam3_video_predictor
        import numpy as np
        import torch
        from pyalicevision import image as avimg
        from PIL import Image
        import json

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

            gpus_to_use = [torch.cuda.current_device()]
            video_predictor = build_sam3_video_predictor(checkpoint_path=chunk.node.segmentationModelPath.evalValue, gpus_to_use=gpus_to_use)

            metadata_deep_model = {}
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelName"] = "SegmentAnything"
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelVersion"] = "sam3-Video-TextPrompt"

            pil_images = []
            mask_images = []

            colorPalette = image.paletteGenerator()
            firstFrameId = chunk_image_paths[0][2]
            frameNumber = len(chunk_image_paths)

            frameIdxToTextPrompt = [0]
            max_frame_num_to_track = None
            track_dir = "forward"
            if chunk.node.timeSlicing.value:
                max_frame_num_to_track = chunk.node.sliceSize.value
                currFrameToTextPrompt = 0
                while currFrameToTextPrompt + chunk.node.sliceSize.value < frameNumber:
                    currFrameToTextPrompt += chunk.node.sliceSize.value
                    frameIdxToTextPrompt.append(currFrameToTextPrompt)
            if chunk.node.combineFwdAndBwdSeg.value:
                track_dir = "both"
                if frameIdxToTextPrompt[-1] < frameNumber - 1:
                    frameIdxToTextPrompt.append(frameNumber - 1)

            logger.info(f"frameIdxToTextPrompt: {frameIdxToTextPrompt}; direction = {track_dir}")

            for idx, path in enumerate(chunk_image_paths):
                img, h_ori, w_ori, PAR, orientation = image.loadImage(str(chunk_image_paths[idx][0]), True)
                pil_images.append(Image.fromarray((255.0 * img).astype("uint8")))
                sourceInfo = {"h_ori": h_ori, "w_ori": w_ori, "PAR": PAR, "orientation": orientation}
                mask_images.append(np.zeros_like(img))

                if firstFrameId is None or chunk_image_paths[idx][2] is None:
                    frameId = idx
                else:
                    frameId = chunk_image_paths[idx][2] - firstFrameId

            response = video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=pil_images,
                    )
            )
            session_id = response["session_id"]

            boxes = {}
            metadata_boxes = {}
            for frameId in range(frameNumber):
                metadata_boxes[frameId] = {}

            for textPrompt in self.text_prompts:

                logger.info(f"textPrompt: {textPrompt}")
                boxes[textPrompt] = {"forward": {}, "backward": {}, "merged": {}}
                cryptoName = "object" if textPrompt == "" else textPrompt
                metadata_deep_model["Meshroom:mrSegmentation:Prompt"] = textPrompt
                for frameId in range(frameNumber):
                    metadata_boxes[frameId][textPrompt] = {"forward": {}, "backward": {}, "merged": {}}

                video_predictor.handle_request(request=dict(type="reset_session", session_id=session_id))

                outputs_per_frame = {}

                prev_overlap_detections_fwd = {}
                next_global_id_fwd: int = 0
                prev_overlap_detections_bwd = {}
                next_global_id_bwd: int = 0
                prev_overlap_detections_merged = {}
                next_global_id_merged: int = 0

                for n, fIdx in enumerate(frameIdxToTextPrompt):

                    logger.info(f"text prompt at relative frame {fIdx} (absolute frame {int(firstFrameId) + fIdx})")

                    video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=fIdx,
                            text=textPrompt,
                        )
                    )
                    outputs_per_frame[fIdx] = sam3Utils.propagateInVideo(video_predictor, session_id, fIdx, max_frame_num_to_track, track_dir)

                    if n == 0:
                        fwd_only = sam3Utils.prepareMasksForVisualization(outputs_per_frame[fIdx])
                        bwd_only = sam3Utils.prepareMasksForVisualization(outputs_per_frame[fIdx])

                        # Initialize prev_overlap_detections_bwd with global IDs
                        # by running a trivial assign_global_ids on just frame 0
                        bwd_frame0_only = {fIdx: bwd_only[fIdx]}
                        _, prev_overlap_detections_bwd, next_global_id_bwd = sam3Utils.assign_global_ids(
                            bwd_frame0_only,
                            {},
                            next_global_id_bwd,
                            similatity_threshold=0.4,
                            use_mask=True,
                        )
                        colorPalette.generate_palette(next_global_id_bwd + 1)
                        logger.info(f"next_global_id_bwd = {next_global_id_bwd}")

                        # Initialize merged tracking for n == 0 as well
                        if chunk.node.combineFwdAndBwdSeg.value:
                            merged_frame0_only = {fIdx: fwd_only[fIdx]}
                            fwd_bwd, prev_overlap_detections_merged, next_global_id_merged = sam3Utils.assign_global_ids(
                                merged_frame0_only,
                                {},
                                next_global_id_merged,
                                similatity_threshold=0.4,
                                use_mask=True,
                            )
                            colorPalette.generate_palette(next_global_id_merged + 1)
                            logger.info(f"next_global_id_merged = {next_global_id_merged}")

                    else:
                        track_fwd = sam3Utils.prepareMasksForVisualization(outputs_per_frame[fIdx])
                        firstFrame = fIdx
                        lastFrame = fIdx if n == len(frameIdxToTextPrompt) - 1 else frameIdxToTextPrompt[n + 1]
                        fwd = {k: v for k,v in track_fwd.items() if k >= firstFrame and k <= lastFrame}

                        fwd_only, prev_overlap_detections_fwd, next_global_id_fwd = sam3Utils.assign_global_ids(
                            fwd,
                            prev_overlap_detections_fwd,
                            next_global_id_fwd,
                            similatity_threshold=0.4,
                            use_mask=True,
                        )
                        colorPalette.generate_palette(next_global_id_fwd + 1)
                        logger.info(f"next_global_id_fwd = {next_global_id_fwd}")

                        if chunk.node.combineFwdAndBwdSeg.value:

                            track_bwd = sam3Utils.prepareMasksForVisualization(outputs_per_frame[fIdx])

                            firstFrame = frameIdxToTextPrompt[n - 1]
                            lastFrame = fIdx
                            bwd = {k: v for k,v in track_bwd.items() if k >= firstFrame and k <= lastFrame}

                            bwd_only, prev_overlap_detections_bwd, next_global_id_bwd = sam3Utils.assign_global_ids(
                                bwd,
                                prev_overlap_detections_bwd,
                                next_global_id_bwd,
                                similatity_threshold=0.4,
                                use_mask=True,
                            )
                            colorPalette.generate_palette(next_global_id_bwd + 1)
                            logger.info(f"next_global_id_bwd = {next_global_id_bwd}")

                            # Create FRESH copies for merge_tracks since the previous calls cleared the dicts
                            track_fwd_for_merge = sam3Utils.prepareMasksForVisualization(outputs_per_frame[frameIdxToTextPrompt[n - 1]])
                            track_bwd_for_merge = sam3Utils.prepareMasksForVisualization(outputs_per_frame[fIdx])
                            firstFrame = frameIdxToTextPrompt[n - 1]
                            lastFrame = fIdx
                            fwd_for_merge = {k: v for k,v in track_fwd_for_merge.items() if k >= firstFrame and k <= lastFrame}
                            bwd_for_merge = {k: v for k,v in track_bwd_for_merge.items() if k >= firstFrame and k <= lastFrame}

                            fwd_bwd, prev_overlap_detections_merged, next_global_id_merged = sam3Utils.merge_tracks(
                                fwd_results=fwd_for_merge,
                                bwd_results=bwd_for_merge,
                                prev_overlap_detections=prev_overlap_detections_merged,
                                next_global_id=next_global_id_merged,
                                similatity_threshold_merge=0.3,
                                similatity_threshold_track=0.4,
                                use_mask=True,
                            )
                            colorPalette.generate_palette(next_global_id_merged + 1)
                            logger.info(f"next_global_id_merged = {next_global_id_merged}")

                        del outputs_per_frame[frameIdxToTextPrompt[n - 1]]

                    # write Fwd from fIdx to frameIdxToTextPrompt[n + 1]
                    lastFIdxFwd = frameIdxToTextPrompt[n + 1] if n < len(frameIdxToTextPrompt) - 1 else frameNumber

                    logger.debug(f"Extract boxes for frame Fwd from : {fIdx} to {lastFIdxFwd - 1}")

                    for frameId in range(fIdx, lastFIdxFwd):
                        colorMaskImageFwd = np.zeros_like(img)
                        if chunk.node.outputCryptomatte.value:
                            crypto_id_fwd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                            crypto_cov_fwd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                            manifest_fwd = {}
                        boxes[textPrompt]["forward"][firstFrameId + frameId] = {}
                        for key, maskBoxProb in fwd_only[frameId].items():
                            mask = maskBoxProb["mask"]
                            mask_images[frameId][mask] = [(int(key) + 1) * 255, 255, 255]
                            color = colorPalette.at(int(key)) if colorPalette.at(int(key)) is not None else [255, 255, 255]
                            colorMaskImageFwd[mask] = [x / 255.0 for x in color]

                            if chunk.node.outputCryptomatte.value:
                                obj_name = f"{cryptoName}_fwd_{int(key)}"
                                f32_hash, hex_val, _ = image.hash_name(obj_name)
                                manifest_fwd[obj_name] = hex_val
                                crypto_id_fwd[mask] = f32_hash
                                crypto_cov_fwd[mask] = 1.0

                            bbox = sam3Utils.xywhNorm2xyxy(maskBoxProb["box_xywh"], sourceInfo["w_ori"], sourceInfo["h_ori"]) # (x, y, x+w, y+h)
                            boxes[textPrompt]["forward"][firstFrameId + frameId][key] = bbox
                            x1, y1, x2, y2 = bbox
                            bbox_str = str(x1) + ";" + str(y1) + ";" + str(x2) + ";" + str(y2)
                            metadata_boxes[frameId][textPrompt]["forward"]["fwd_" + textPrompt + "_" + str(key)] = bbox_str

                        if chunk.node.outputColorMasks.value:
                            if chunk.node.keepFilename.value:
                                outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + textPrompt + "_fwd_" + str(Path(chunk_image_paths[frameId][0]).stem) + ".exr")
                            else:
                                outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + textPrompt + "_fwd_" + str(chunk_image_paths[frameId][1]) + ".exr")

                            optWrite = avimg.ImageWriteOptions()
                            optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)

                            image.writeImage(outputFileColorMask, colorMaskImageFwd, sourceInfo["h_ori"], sourceInfo["w_ori"], sourceInfo["orientation"], sourceInfo["PAR"], metadata_deep_model, optWrite)

                        if chunk.node.outputCryptomatte.value:
                            if chunk.node.keepFilename.value:
                                cryptomattePath = os.path.join(chunk.node.output.value, "cryptomatte_" + textPrompt + "_fwd_" + str(Path(chunk_image_paths[frameId][0]).stem) + ".exr")
                            else:
                                cryptomattePath = os.path.join(chunk.node.output.value, "cryptomatte_" + textPrompt + "_fwd_" + str(chunk_image_paths[frameId][1]) + ".exr")

                            image.writeCryptomatte(cryptomattePath, cryptoName, img.shape[1], img.shape[0], manifest_fwd, crypto_id_fwd, crypto_cov_fwd)

                    if chunk.node.combineFwdAndBwdSeg.value:

                        # write Bwd from frameIdxToTextPrompt[n - 1] to fIdx
                        firstFIdxBwd = frameIdxToTextPrompt[n - 1] + 1 if n > 0 else fIdx
                        for frameId in range(firstFIdxBwd, fIdx + 1):
                            colorMaskImageBwd = np.zeros_like(img)
                            if chunk.node.outputCryptomatte.value:
                                crypto_id_bwd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                                crypto_cov_bwd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                                manifest_bwd = {}
                            boxes[textPrompt]["backward"][firstFrameId + frameId] = {}
                            for key, maskBoxProb in bwd_only[frameId].items():
                                mask = maskBoxProb["mask"]
                                mask_images[frameId][mask] = [(int(key) + 1) * 255, 255, 255]
                                color = colorPalette.at(int(key)) if colorPalette.at(int(key)) is not None else [255, 255, 255]
                                colorMaskImageBwd[mask] = [x / 255.0 for x in color]
                                if chunk.node.outputCryptomatte.value:
                                    obj_name = f"{cryptoName}_bwd_{int(key)}"
                                    f32_hash, hex_val, _ = image.hash_name(obj_name)
                                    manifest_bwd[obj_name] = hex_val
                                    crypto_id_bwd[mask] = f32_hash
                                    crypto_cov_bwd[mask] = 1.0
                                bbox = sam3Utils.xywhNorm2xyxy(maskBoxProb["box_xywh"], sourceInfo["w_ori"], sourceInfo["h_ori"]) # (x, y, x+w, y+h)
                                boxes[textPrompt]["backward"][firstFrameId + frameId][key] = bbox
                                x1,y1,x2,y2 = bbox
                                bbox_str = str(x1) + ";" + str(y1) + ";" + str(x2) + ";" + str(y2)
                                metadata_boxes[frameId][textPrompt]["backward"]["bwd_" + textPrompt + "_" + str(key)] = bbox_str

                            if chunk.node.outputColorMasks.value:
                                if chunk.node.keepFilename.value:
                                    outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + textPrompt + "_bwd_" + str(Path(chunk_image_paths[frameId][0]).stem) + ".png")
                                else:
                                    outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + textPrompt + "_bwd_" + str(chunk_image_paths[frameId][1]) + ".png")

                                optWrite = avimg.ImageWriteOptions()
                                optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)

                                image.writeImage(outputFileColorMask, colorMaskImageBwd, sourceInfo["h_ori"], sourceInfo["w_ori"], sourceInfo["orientation"], sourceInfo["PAR"], metadata_deep_model, optWrite)

                            if chunk.node.outputCryptomatte.value:
                                if chunk.node.keepFilename.value:
                                    cryptomattePath = os.path.join(chunk.node.output.value, "cryptomatte_" + textPrompt + "_bwd_" + str(Path(chunk_image_paths[frameId][0]).stem) + ".exr")
                                else:
                                    cryptomattePath = os.path.join(chunk.node.output.value, "cryptomatte_" + textPrompt + "_bwd_" + str(chunk_image_paths[frameId][1]) + ".exr")

                                image.writeCryptomatte(cryptomattePath, cryptoName, img.shape[1], img.shape[0], manifest_bwd, crypto_id_bwd, crypto_cov_bwd)

                        if n > 0:
                            for frameId in range(firstFIdxBwd - 1, fIdx + 1):
                                colorMaskImageMerged = np.zeros_like(img)
                                boxes[textPrompt]["merged"][firstFrameId + frameId] = {}
                                for key, maskBoxProb in fwd_bwd[frameId].items():
                                    mask = maskBoxProb["mask"]
                                    mask_images[frameId][mask] = [(int(key) + 1) * 255, 255, 255]
                                    color = colorPalette.at(int(key)) if colorPalette.at(int(key)) is not None else [255, 255, 255]
                                    colorMaskImageMerged[mask] = [x / 255.0 for x in color]
                                    
                                    bbox = sam3Utils.xywhNorm2xyxy(maskBoxProb["box_xywh"], sourceInfo["w_ori"], sourceInfo["h_ori"]) # (x, y, x+w, y+h)
                                    boxes[textPrompt]["merged"][firstFrameId + frameId][key] = bbox
                                    x1,y1,x2,y2 = bbox
                                    bbox_str = str(x1) + ";" + str(y1) + ";" + str(x2) + ";" + str(y2)
                                    metadata_boxes[frameId][textPrompt]["merged"]["merged_" + textPrompt + "_" + str(key)] = bbox_str

                                if chunk.node.outputColorMasks.value:
                                    if chunk.node.keepFilename.value:
                                        outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + textPrompt + "_merged_" + str(Path(chunk_image_paths[frameId][0]).stem) + ".exr")
                                    else:
                                        outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + textPrompt + "_merged_" + str(chunk_image_paths[frameId][1]) + ".exr")

                                    optWrite = avimg.ImageWriteOptions()
                                    optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)

                                    image.writeImage(outputFileColorMask, colorMaskImageMerged, sourceInfo["h_ori"], sourceInfo["w_ori"], sourceInfo["orientation"], sourceInfo["PAR"], metadata_deep_model, optWrite)

            prompts = [textPrompt.strip() for textPrompt in self.text_prompts if textPrompt.strip()]
            metadata_deep_model["Meshroom:mrSegmentation:Prompt"] = ";".join(prompts)

            for frameId in range(frameNumber):
                if chunk.node.maskInvert.value:
                    mask = (mask_images[frameId][:,:,0:1] == 0).astype('float32')
                else:
                    mask = (mask_images[frameId][:,:,0:1] > 0).astype('float32')
                logger.info("frameId: {} - {}".format(frameId, chunk_image_paths[frameId][0]))

                if chunk.node.keepFilename.value:
                    outputFileMask = os.path.join(chunk.node.output.value, Path(chunk_image_paths[frameId][0]).stem + "." + chunk.node.extensionOut.value)
                else:
                    outputFileMask = os.path.join(chunk.node.output.value, str(chunk_image_paths[frameId][1]) + "." + chunk.node.extensionOut.value)

                optWrite = avimg.ImageWriteOptions()
                optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
                if Path(outputFileMask).suffix.lower() == ".exr":
                    optWrite.exrCompressionMethod(avimg.EImageExrCompression_stringToEnum("DWAA"))
                    optWrite.exrCompressionLevel(300)

                frame_metadata_deep_model = dict(metadata_deep_model)
                for prompt in self.text_prompts:
                    for direction in ["forward", "backward", "merged"]:
                        for k, box in metadata_boxes[frameId][prompt][direction].items():
                            frame_metadata_deep_model["Meshroom:mrSegmentation:" + k] = box

                image.writeImage(outputFileMask, mask, sourceInfo["h_ori"], sourceInfo["w_ori"], sourceInfo["orientation"], sourceInfo["PAR"], frame_metadata_deep_model, optWrite)

            jsonFilename = chunk.node.output.value + "/bboxes.json"
            with open(jsonFilename, "w", encoding="utf_8") as f:
                json.dump(boxes, f, indent=4, ensure_ascii=False)

            video_predictor.handle_request(request=dict(type="close_session", session_id=session_id))

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
