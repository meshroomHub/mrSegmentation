__version__ = "0.1"

import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

import logging
logger = logging.getLogger("VideoSegmentationSam3Text")

class Sam3VideoNodeSize(desc.MultiDynamicNodeSize):
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
        
class VideoSegmentationSam3Text(desc.Node):
    size = Sam3VideoNodeSize(['input', 'extensionIn'])
    gpu = desc.Level.EXTREME

    category = "Segmentation"
    documentation = """
Based on the Segment Anything video predictor model 3, the node generates a binary mask, a colored mask and an exr cryptomatte
from a text prompt.
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
            label="Time Slicing",
            description="Enable time slicing by adding text prompt every N frames and Propagating the masks over N frames.\n"
                        "Propagation is forward only by default, or both forward and backward when 'Combine Forward and Backward Segmentation'\n"
                        "is enabled.",
            value=False,
        ),
        desc.IntParam(
            name="sliceSize",
            label="Slice Size",
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
            label="Output Cryptomatte",
            description="Generate exr images containing cryptomatte to encode the segmentation results.",
            value=False,
        ),
        desc.BoolParam(
            name="outputColorMasks",
            label="Output Color Masks",
            description="Generate colored masks where colors are linked with object Ids.",
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
            name="colorMasksFwd",
            label="Colored Masks Forward",
            description="Colored segmentation masks resulting from forward propagation. Colors correspond to instance indexes.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputColorMasks.value,
            group="",
        ),
        desc.File(
            name="colorMasksBwd",
            label="Colored Masks Backward",
            description="Colored segmentation masks resulting from backward propagation. Colors correspond to instance indexes.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputColorMasks.value and node.combineFwdAndBwdSeg.value,
            group="",
        ),
        desc.File(
            name="cryptomatteFwd",
            label="Cryptomatte Forward",
            description="Cryptomatte resulting from forward propagation embedded in exr images.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputCryptomatte.value,
            group="",
        ),
        desc.File(
            name="cryptomatteBwd",
            label="Cryptomatte Backward",
            description="Cryptomatte resulting from backward propagation embedded in exr images.",
            semantic="image",
            value=None,
            enabled=lambda node: node.outputCryptomatte.value and node.combineFwdAndBwdSeg.value,
            group="",
        ),
    ]

    def preprocess(self, node):
        import re
        extension = node.extensionIn.value
        input_path = node.input.value
        image_paths = get_image_paths_list(input_path, extension)
        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {input_path}')
        self.image_paths = image_paths
        if node.prompt.value == "":
            raise ValueError(f'Text prompt is empty')
        self.textPrompts = re.split(r'[\n]+', node.prompt.value)
        self.textPrompts = [str(textPrompt) for textPrompt in self.textPrompts if textPrompt]
        srcFilename = "<FILESTEM>" if node.keepFilename.value else "<VIEW_ID>"
        node.colorMasksFwd.value = node.output.value + "/colorMask_" + self.textPrompts[0] + "_fwd_" + srcFilename + ".png"
        node.colorMasksBwd.value = node.output.value + "/colorMask_" + self.textPrompts[0] + "_bwd_" + srcFilename + ".png"
        node.cryptomatteFwd.value = node.output.value + "/cryptomatte_" + self.textPrompts[0] + "_fwd_" + srcFilename + ".png"
        node.cryptomatteBwd.value = node.output.value + "/cryptomatte_" + self.textPrompts[0] + "_bwd_" + srcFilename + ".png"

    def processChunk(self, chunk):
        from segmentationRDS import image, sam3Utils
        from sam3.model_builder import build_sam3_video_predictor
        import numpy as np
        import torch
        from pyalicevision import image as avimg
        from PIL import Image
        import json

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

            metadata_deep_model = {}
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelName"] = "SegmentAnything"
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelVersion"] = "sam3-Video"

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
                pil_images.append(Image.fromarray((255.0*img).astype("uint8")))
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

            for textPrompt in self.textPrompts:

                logger.info(f"textPrompt: {textPrompt}")
                boxes[textPrompt] = {"forward": {}, "backward": {}}
                cryptoName = "object" if textPrompt == "" else textPrompt

                video_predictor.handle_request(request=dict(type="reset_session", session_id=session_id))

                outputs_per_frame = {}
                outputs_per_frame_fwd = {}
                outputs_per_frame_bwd = {}
                max_obj_id = 0
                for n, fIdx in enumerate(frameIdxToTextPrompt):
                    video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=fIdx,
                            text=textPrompt,
                        )
                    )
                    outputs_per_frame[fIdx] = sam3Utils.propagateInVideo(video_predictor, session_id, fIdx, max_frame_num_to_track, track_dir)

                    max_obj_id_at_fIdx = 0
                    for f, obj in outputs_per_frame[fIdx].items():
                        max_obj_id_at_fIdx_f = -1
                        if len(obj["out_obj_ids"].tolist()) > 0:
                            max_obj_id_at_fIdx_f = max(obj["out_obj_ids"].tolist())
                        if max_obj_id_at_fIdx_f > max_obj_id_at_fIdx:
                            max_obj_id_at_fIdx = max_obj_id_at_fIdx_f
                    if max_obj_id_at_fIdx > max_obj_id:
                        max_obj_id = max_obj_id_at_fIdx

                    logger.debug(f"max_obj_id = {max_obj_id}")

                    colorPalette.generate_palette(max_obj_id + 1)

                    if n == 0:
                        outputs_per_frame_fwd[fIdx] = outputs_per_frame[fIdx]
                        outputs_per_frame_bwd[fIdx] = outputs_per_frame[fIdx]
                    else:

                        logger.debug(f"Inputs for mapping at key frame {fIdx}")
                        logger.debug(f"Detected Objects at frame {fIdx}:")
                        sam3Utils.displayAt(outputs_per_frame, fIdx, fIdx, sourceInfo["w_ori"], sourceInfo["h_ori"], logger)
                        logger.debug(f"Propagated Objects at frame {fIdx} from frame {frameIdxToTextPrompt[n - 1]}:")
                        sam3Utils.displayAt(outputs_per_frame_fwd, frameIdxToTextPrompt[n - 1], fIdx, sourceInfo["w_ori"], sourceInfo["h_ori"], logger)

                        mapping_fwd = sam3Utils.mapIds(outputs_per_frame[fIdx][fIdx], outputs_per_frame_fwd[frameIdxToTextPrompt[n - 1]][fIdx],
                                                       sourceInfo["w_ori"], sourceInfo["h_ori"], logger)

                        logger.debug(f"mapping fwd at key frame {fIdx}:\n{mapping_fwd}")

                        outputs_per_frame_fwd[fIdx] = sam3Utils.updateSam3ObjectIds(outputs_per_frame[fIdx], mapping_fwd)

                        del outputs_per_frame_fwd[frameIdxToTextPrompt[n - 1]]

                        logger.debug(f"Output after mapping fwd has been applied at key frame {fIdx}")
                        sam3Utils.displayAt(outputs_per_frame_fwd, fIdx, fIdx, sourceInfo["w_ori"], sourceInfo["h_ori"], logger)

                        if chunk.node.combineFwdAndBwdSeg.value:

                            logger.debug(f"Inputs for mapping at key frame {frameIdxToTextPrompt[n - 1]}")
                            logger.debug(f"Detected Objects at frame {frameIdxToTextPrompt[n - 1]}:")
                            sam3Utils.displayAt(outputs_per_frame_bwd, frameIdxToTextPrompt[n - 1], frameIdxToTextPrompt[n - 1], sourceInfo["w_ori"], sourceInfo["h_ori"], logger)
                            logger.debug(f"Propagated Objects at frame {frameIdxToTextPrompt[n - 1]} from frame {fIdx}:")
                            sam3Utils.displayAt(outputs_per_frame, fIdx, frameIdxToTextPrompt[n - 1], sourceInfo["w_ori"], sourceInfo["h_ori"], logger)

                            mapping_bwd = sam3Utils.mapIds(outputs_per_frame[fIdx][frameIdxToTextPrompt[n - 1]],
                                                           outputs_per_frame_bwd[frameIdxToTextPrompt[n - 1]][frameIdxToTextPrompt[n - 1]],
                                                           sourceInfo["w_ori"], sourceInfo["h_ori"], logger)

                            logger.debug(f"mapping bwd at key frame {frameIdxToTextPrompt[n - 1]}:\n{mapping_bwd}")

                            outputs_per_frame_bwd[fIdx] = sam3Utils.updateSam3ObjectIds(outputs_per_frame[fIdx], mapping_bwd)

                            del outputs_per_frame_bwd[frameIdxToTextPrompt[n - 1]]

                            logger.debug(f"Output after mapping bwd has been applied at key frame {fIdx}")
                            sam3Utils.displayAt(outputs_per_frame_bwd, fIdx, fIdx, sourceInfo["w_ori"], sourceInfo["h_ori"], logger)

                        del outputs_per_frame[frameIdxToTextPrompt[n - 1]]


                    logger.debug(f"Keys: {outputs_per_frame_fwd[fIdx].keys()}")

                    # write Fwd from fIdx to frameIdxToTextPrompt[n + 1]
                    lastFIdxFwd = frameIdxToTextPrompt[n + 1] if n < len(frameIdxToTextPrompt) - 1 else fIdx + 1
                    outputs_per_frame_visu = sam3Utils.prepareMasksForVisualization(outputs_per_frame_fwd[fIdx])

                    logger.debug(f"Extract boxes for frame Fwd from : {fIdx} to {lastFIdxFwd - 1}")

                    for frameId in range(fIdx, lastFIdxFwd):
                        colorMaskImageFwd = np.zeros_like(img)
                        if chunk.node.outputCryptomatte.value:
                            crypto_id_fwd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                            crypto_cov_fwd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                            manifest_fwd = {}
                        boxes[textPrompt]["forward"][frameId] = {}
                        for key, maskBoxProb in outputs_per_frame_visu[frameId].items():
                            mask = maskBoxProb["mask"]
                            mask_images[frameId][mask] = [255, 255, 255]
                            color = colorPalette.at(int(key)) if colorPalette.at(int(key)) is not None else [255, 255, 255]
                            colorMaskImageFwd[mask] = [x/255.0 for x in color]

                            if chunk.node.outputCryptomatte.value:
                                obj_name = f"{cryptoName}_fwd_{int(key)}"
                                f32_hash, hex_val, _ = image.hash_name(obj_name)
                                manifest_fwd[obj_name] = hex_val
                                crypto_id_fwd[mask] = f32_hash
                                crypto_cov_fwd[mask] = 1.0

                            bbox = sam3Utils.xywhNorm2xyxy(maskBoxProb["box_xywh"], sourceInfo["w_ori"], sourceInfo["h_ori"]) # (x, y, x+w, y+h)
                            boxes[textPrompt]["forward"][frameId][key] = bbox

                        if chunk.node.outputColorMasks.value:
                            if chunk.node.keepFilename.value:
                                outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + textPrompt + "_fwd_" + str(Path(chunk_image_paths[frameId][0]).stem) + ".png")
                            else:
                                outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + textPrompt + "_fwd_" + str(chunk_image_paths[frameId][1]) + ".png")

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
                        outputs_per_frame_visu = sam3Utils.prepareMasksForVisualization(outputs_per_frame_bwd[fIdx])
                        for frameId in range(firstFIdxBwd, fIdx + 1):
                            colorMaskImageBwd = np.zeros_like(img)
                            if chunk.node.outputCryptomatte.value:
                                crypto_id_bwd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                                crypto_cov_bwd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                                manifest_bwd = {}
                            boxes[textPrompt]["backward"][frameId] = {}
                            for key, maskBoxProb in outputs_per_frame_visu[frameId].items():
                                mask = maskBoxProb["mask"]
                                mask_images[frameId][mask] = [255, 255, 255]
                                color = colorPalette.at(int(key)) if colorPalette.at(int(key)) is not None else [255, 255, 255]
                                colorMaskImageBwd[mask] = [x/255.0 for x in color]
                                if chunk.node.outputCryptomatte.value:
                                    obj_name = f"{cryptoName}_bwd_{int(key)}"
                                    f32_hash, hex_val, _ = image.hash_name(obj_name)
                                    manifest_bwd[obj_name] = hex_val
                                    crypto_id_bwd[mask] = f32_hash
                                    crypto_cov_bwd[mask] = 1.0
                                bbox = sam3Utils.xywhNorm2xyxy(maskBoxProb["box_xywh"], sourceInfo["w_ori"], sourceInfo["h_ori"]) # (x, y, x+w, y+h)
                                boxes[textPrompt]["backward"][frameId][key] = bbox

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

                    image.writeImage(outputFileMask, mask, sourceInfo["h_ori"], sourceInfo["w_ori"], sourceInfo["orientation"], sourceInfo["PAR"], metadata_deep_model, optWrite)

            jsonFilename = chunk.node.output.value + "/bboxes.json"
            with open(jsonFilename, "w", encoding="utf_8") as f:
                json.dump(boxes, f, indent=4, ensure_ascii=False)

            video_predictor.handle_request(request=dict(type="close_session", session_id=session_id))

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