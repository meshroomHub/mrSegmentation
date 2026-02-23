__version__ = "0.1"

import os
from pathlib import Path
import struct

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
            name="combineFwdAndBwdSeg",
            label="Combine Forward and Backward Segmentation",
            description="Launch segmentation in both forward and backward directions and combine masks.",
            value=False,
        ),
        desc.BoolParam(
            name="timeSlicing",
            label="Time Slicing",
            description="Enable time slicing by adding text prompt every N frames and by propagating forward on N frames.",
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
            enabled=lambda node: not node.timeSlicing.value,
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
            value=lambda attr: "{nodeCacheFolder}/cryptomatte_fwd_" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + ".exr",
            enabled=lambda node: node.outputCryptomatte.value,
            group="",
        ),
        desc.File(
            name="cryptomatteBwd",
            label="Cryptomatte Backward",
            description="Cryptomatte resulting from backward propagation embedded in exr images.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/cryptomatte_bwd_" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + ".exr",
            enabled=lambda node: node.outputCryptomatte.value and node.combineFwdAndBwdSeg.value,
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

    def propagate_in_video(self, predictor, session_id, start_frame_idx=None, max_frame_num_to_track=None, direction="both"):
        # we will just propagate from frame 0 to the end of the video
        outputs_per_frame = {}
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction=direction,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
            )
        ):
            outputs_per_frame[response["frame_index"]] = response["outputs"]
        return outputs_per_frame
        
    def hash_name(self, name):
        import mmh3
        import numpy as np
        hash_32 = mmh3.hash(name, seed=0) & 0xFFFFFFFF
        f32_val = np.frombuffer(struct.pack('<I', hash_32), dtype=np.float32)[0]
        f32_hex = hex(struct.unpack('<I', struct.pack('<f', f32_val))[0])[2:]
        return f32_val, f32_hex, hash_32

    def save_cryptomatte(self, filepath, crypto_name, w, h, manifest, crypto_id, crypto_cov):
        import OpenImageIO as oiio
        import json
        import numpy as np

        spec = oiio.ImageSpec(w, h, 7, oiio.FLOAT)
        spec.channelnames = (crypto_name+".red", crypto_name+".green", crypto_name+".blue",
                            crypto_name+"00.red", crypto_name+"00.green", crypto_name+"00.blue", crypto_name+"00.alpha")
        _, _, h32 = self.hash_name(crypto_name)
        crypto_key = f"{h32 & 0xFFFFFFFF:08x}"[:7]
        spec.attribute(f"cryptomatte/{crypto_key}/name", crypto_name)
        spec.attribute(f"cryptomatte/{crypto_key}/manifest", json.dumps(manifest))
        spec.attribute(f"cryptomatte/{crypto_key}/hash", "MurmurHash3_32")
        spec.attribute(f"cryptomatte/{crypto_key}/conversion", "uint32_to_float32")

        crypto_zeros = np.zeros((h, w), dtype=np.float32)
        cryptomatteImg = oiio.ImageOutput.create(str(filepath))
        cryptomatteImg.open(filepath, spec)
        cryptomatte_data = np.dstack((crypto_zeros, crypto_zeros, crypto_zeros, crypto_id, crypto_cov, crypto_zeros, crypto_zeros))
        cryptomatteImg.write_image(cryptomatte_data)
        cryptomatteImg.close()

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
        self.textPrompts = [textPrompt for textPrompt in self.textPrompts if textPrompt]
        node.colorMasksFwd.value = node.output.value + "/colorMask_" + self.textPrompts[0] + "_fwd_" + ("<FILESTEM>" if node.keepFilename.value else "<VIEW_ID>") + ".png"
        node.colorMasksBwd.value = node.output.value + "/colorMask_" + self.textPrompts[0] + "_bwd_" + ("<FILESTEM>" if node.keepFilename.value else "<VIEW_ID>") + ".png"

    def processChunk(self, chunk):
        import re
        from segmentationRDS import image
        from sam3.model_builder import build_sam3_video_predictor
        import numpy as np
        import torch
        from pyalicevision import image as avimg
        from PIL import Image

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
            frameIdxToTextPrompt_fwd = [0]
            frameIdxToTextPrompt_bwd = [frameNumber - 1]
            max_frame_num_to_track_fwd = None
            max_frame_num_to_track_bwd = None
            if chunk.node.timeSlicing.value:
                if chunk.node.sliceSize.value > 0 and chunk.node.sliceSize.value <= frameNumber:
                    currFrameToTextPrompt_fwd = 0
                    currFrameToTextPrompt_bwd = frameNumber - 1
                    max_frame_num_to_track_fwd = chunk.node.sliceSize.value - 1
                    max_frame_num_to_track_bwd = chunk.node.sliceSize.value
                    while currFrameToTextPrompt_fwd + chunk.node.sliceSize.value < frameNumber:
                        currFrameToTextPrompt_fwd += chunk.node.sliceSize.value
                        frameIdxToTextPrompt_fwd.append(currFrameToTextPrompt_fwd)
                    while currFrameToTextPrompt_bwd - chunk.node.sliceSize.value >= 0:
                        currFrameToTextPrompt_bwd -= chunk.node.sliceSize.value
                        frameIdxToTextPrompt_bwd.append(currFrameToTextPrompt_bwd)

            logger.debug(f"frameIdxToTextPromptFwd: {frameIdxToTextPrompt_fwd}")
            logger.debug(f"frameIdxToTextPromptBwd: {frameIdxToTextPrompt_bwd}")
            
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

            for textPrompt in self.textPrompts:

                video_predictor.handle_request(request=dict(type="reset_session", session_id=session_id))

                outputs_per_frame_fwd = {}
                for n, fIdx in enumerate(frameIdxToTextPrompt_fwd):
                    video_predictor.handle_request(
                        request=dict(
                            type="add_prompt",
                            session_id=session_id,
                            frame_index=fIdx,
                            text=textPrompt,
                        )
                    )
                    outputs_per_frame_curr_fwd = self.propagate_in_video(video_predictor, session_id, fIdx, max_frame_num_to_track_fwd, "forward")
                    outputs_per_frame_fwd.update(outputs_per_frame_curr_fwd)

                logger.debug(f"Fwd keys: {outputs_per_frame_fwd.keys()}")

                video_predictor.handle_request(request=dict(type="reset_session", session_id=session_id))

                outputs_per_frame_bwd = {}
                if chunk.node.combineFwdAndBwdSeg.value:
                    for n, fIdx in enumerate(frameIdxToTextPrompt_bwd):
                        video_predictor.handle_request(
                            request=dict(
                                type="add_prompt",
                                session_id=session_id,
                                frame_index=fIdx,
                                text=textPrompt,
                            )
                        )
                        outputs_per_frame_curr_bwd = self.propagate_in_video(video_predictor, session_id, fIdx, max_frame_num_to_track_bwd, "backward")
                        outputs_per_frame_bwd.update(outputs_per_frame_curr_bwd)
                    logger.debug(f"Bwd keys: {outputs_per_frame_bwd.keys()}")

                outputs_per_frame_fwd = self.prepare_masks_for_visualization(outputs_per_frame_fwd)
                outputs_per_frame_bwd = self.prepare_masks_for_visualization(outputs_per_frame_bwd)

                for frameId, masks in outputs_per_frame_fwd.items():
                    colorMaskImageFwd = np.zeros_like(img)
                    if chunk.node.outputCryptomatte.value:
                        crypto_id_fwd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                        crypto_cov_fwd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                        manifest_fwd = {}

                    if len(masks.keys()) > 0:
                        colorPalette.generate_palette(max(masks.keys()) + 1)

                    cryptoName = "object" if textPrompt == "" else textPrompt
                    for key, mask in masks.items():
                        mask_images[frameId][mask] = [255, 255, 255]
                        color = colorPalette.at(int(key)) if colorPalette.at(int(key)) is not None else [255, 255, 255]
                        colorMaskImageFwd[mask] = [x/255.0 for x in color]
                        if chunk.node.outputCryptomatte.value:
                            obj_name = f"{cryptoName}_fwd_{int(key)}"
                            f32_hash, hex_val, _ = self.hash_name(obj_name)
                            manifest_fwd[obj_name] = hex_val
                            crypto_id_fwd[mask] = f32_hash
                            crypto_cov_fwd[mask] = 1.0

                    if frameId in outputs_per_frame_bwd.keys():
                        colorMaskImageBwd = np.zeros_like(img)
                        if chunk.node.outputCryptomatte.value:
                            crypto_id_bwd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                            crypto_cov_bwd = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
                            manifest_bwd = {}

                        if len(outputs_per_frame_bwd[frameId].keys()) > 0:
                            colorPalette.generate_palette(max(outputs_per_frame_bwd[frameId].keys()) + 1)

                        for key, mask in outputs_per_frame_bwd[frameId].items():
                            mask_images[frameId][mask] = [255, 255, 255]
                            color = colorPalette.at(int(key)) if colorPalette.at(int(key)) is not None else [255, 255, 255]
                            colorMaskImageBwd[mask] = [x/255.0 for x in color]
                            if chunk.node.outputCryptomatte.value:
                                obj_name = f"{cryptoName}_bwd_{int(key)}"
                                f32_hash, hex_val, _ = self.hash_name(obj_name)
                                manifest_bwd[obj_name] = hex_val
                                crypto_id_bwd[mask] = f32_hash
                                crypto_cov_bwd[mask] = 1.0

                    if chunk.node.outputCryptomatte.value:
                        if chunk.node.keepFilename.value:
                            cryptomattePath = os.path.join(chunk.node.output.value, "cryptomatte_" + textPrompt + "_fwd_" + str(Path(chunk_image_paths[frameId][0]).stem) + ".exr")
                        else:
                            cryptomattePath = os.path.join(chunk.node.output.value, "cryptomatte_" + textPrompt + "_fwd_" + str(chunk_image_paths[frameId][1]) + ".exr")

                        self.save_cryptomatte(cryptomattePath, cryptoName, img.shape[1], img.shape[0], manifest_fwd, crypto_id_fwd, crypto_cov_fwd)

                        if chunk.node.combineFwdAndBwdSeg.value:
                            if chunk.node.keepFilename.value:
                                cryptomattePath = os.path.join(chunk.node.output.value, "cryptomatte_" + textPrompt + "_bwd_" + str(Path(chunk_image_paths[frameId][0]).stem) + ".exr")
                            else:
                                cryptomattePath = os.path.join(chunk.node.output.value, "cryptomatte_" + textPrompt + "_bwd_" + str(chunk_image_paths[frameId][1]) + ".exr")

                            self.save_cryptomatte(cryptomattePath, cryptoName, img.shape[1], img.shape[0], manifest_bwd, crypto_id_bwd, crypto_cov_bwd)

                    if chunk.node.outputColorMasks.value:
                        if chunk.node.keepFilename.value:
                            outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + textPrompt + "_fwd_" + str(Path(chunk_image_paths[frameId][0]).stem) + ".png")
                        else:
                            outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + textPrompt + "_fwd_" + str(chunk_image_paths[frameId][1]) + ".png")

                        optWrite = avimg.ImageWriteOptions()
                        optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)

                        image.writeImage(outputFileColorMask, colorMaskImageFwd, sourceInfo["h_ori"], sourceInfo["w_ori"], sourceInfo["orientation"], sourceInfo["PAR"], metadata_deep_model, optWrite)

                        if chunk.node.combineFwdAndBwdSeg.value:
                            if chunk.node.keepFilename.value:
                                outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + textPrompt + "_bwd_" + str(Path(chunk_image_paths[frameId][0]).stem) + ".png")
                            else:
                                outputFileColorMask = os.path.join(chunk.node.output.value, "colorMask_" + textPrompt + "_bwd_" + str(chunk_image_paths[frameId][1]) + ".png")

                            image.writeImage(outputFileColorMask, colorMaskImageBwd, sourceInfo["h_ori"], sourceInfo["w_ori"], sourceInfo["orientation"], sourceInfo["PAR"], metadata_deep_model, optWrite)

            video_predictor.handle_request(request=dict(type="close_session", session_id=session_id))

            for frameId in outputs_per_frame_fwd.keys():

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