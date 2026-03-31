__version__ = "1.0"

import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
from pyalicevision import parallelization as avpar

import logging
logger = logging.getLogger("VideoSegmentationSam3Boxes")

class VideoSegmentationSam3Boxes(desc.Node):
    size = avpar.DynamicViewsSize("input")
    gpu = desc.Level.EXTREME

    category = "Segmentation"
    documentation = """
Based on the Segment Anything video predictor model 3, the node generates binary masks from a set of
bounding boxes contained in a json file.
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
            label="Inputx2",
            description="Folder containing source images upscale by 2.",
            value="",
        ),
        desc.File(
            name="inputx4",
            label="Inputx4",
            description="Folder containing source images upscale by 4.",
            value="",
        ),
        desc.File(
            name="masksFolder",
            label="Masks Folder",
            description="Folder containing the masks computed at original resolution.",
            value="",
        ),
        desc.File(
            name="bboxesFolder",
            label="Bounding Boxes Folder",
            description="Folder containing the bboxes.json file associated to the sfmData used as input.",
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

    def preprocess(self, node):
        input_path = node.input.value
        image_paths = get_image_paths_list(input_path, node.inputx2.value, node.inputx4.value)
        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {input_path}')
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
            metadata_deep_model["Meshroom:mrSegmentation:DeepModelVersion"] = "sam3-Video-Crop"

            # bboxes.json decoding
            json_path = os.path.join(chunk.node.bboxesFolder.value, "bboxes.json")
            frame_w = chunk_image_paths[0][3]
            frame_h = chunk_image_paths[0][4]
            par = chunk_image_paths[0][5]
            x2_ok = os.path.exists(chunk.node.inputx2.value)
            x4_ok = os.path.exists(chunk.node.inputx4.value)
            bboxes = bboxUtils.extract_tracking(json_path, frame_w, frame_h, x2_ok, x4_ok, par)

            logger.info(f"bboxes.keys() = {bboxes.keys()}")

            full_mask_images = {}
            img, h_ori, w_ori, p_a_r, orientation = image.loadImage(str(chunk_image_paths[0][0]), True)
            sourceInfo = {"h_ori": h_ori, "w_ori": w_ori, "PAR": p_a_r, "orientation": orientation}
            for frameId, image_path in enumerate(chunk_image_paths):
                full_mask_images[image_path[2]] = np.zeros_like(img)

            for key, frame_chunks in bboxes.items():

                if "_" in key:
                    textPrompt, obj_id = key.rsplit('_', 1)
                else:
                    textPrompt, obj_id = key, ""
                logger.info(f"key = {key} ; text prompt = {textPrompt} ; obj_id = {obj_id}")

                for frame_chunk in frame_chunks:
                    logger.info(frame_chunk)
                    pil_images = []
                    firstFrameId = frame_chunk.start_frame
                    for frame_idx, box in sorted(frame_chunk.boxes.items()):
                        x1, y1, x2, y2 = bboxUtils.box_to_display(box, sourceInfo["PAR"])
                        box_w = x2 - x1
                        box_h = y2 - y1

                        if box_w == 252 and box_h == 252:
                            img, h_ori, w_ori, p_a_r, orientation = image.loadImage(str(chunk_image_paths[frame_idx - firstFrameId][7]), True)
                            imgBuf = oiio.ImageBuf(img)
                            imgBuf = oiio.ImageBufAlgo.crop(imgBuf, roi=oiio.ROI(4*x1, 4*x2, 4*y1, 4*y2))
                        elif box_w == 504 and box_h == 504:
                            img, h_ori, w_ori, p_a_r, orientation = image.loadImage(str(chunk_image_paths[frame_idx - firstFrameId][6]), True)
                            imgBuf = oiio.ImageBuf(img)
                            imgBuf = oiio.ImageBufAlgo.crop(imgBuf, roi=oiio.ROI(2*x1, 2*x2, 2*y1, 2*y2))
                        else:
                            img, h_ori, w_ori, p_a_r, orientation = image.loadImage(str(chunk_image_paths[frame_idx - firstFrameId][0]), True)
                            imgBuf = oiio.ImageBuf(img)
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
                    outputs_per_frame = sam3Utils.propagateInVideo(video_predictor, session_id) #, fIdx, max_frame_num_to_track, track_dir)
                    outputs_per_frame_visu = sam3Utils.prepareMasksForVisualization(outputs_per_frame)

                    for frame_idx, box in sorted(frame_chunk.boxes.items()):
                        x1, y1, x2, y2 = box
                        box_w = x2 - x1
                        box_h = y2 - y1
                        frameId = frame_idx - firstFrameId
                        for key, maskBoxProb in outputs_per_frame_visu[frameId].items():
                            mask = maskBoxProb["mask"]
                            buf_in = oiio.ImageBuf(mask.astype('float32'))
                            buf_out = oiio.ImageBufAlgo.resample(buf_in, roi=oiio.ROI(0, box_w, 0, box_h))
                            mask = buf_out.get_pixels().reshape(box_h, box_w, 1)
                            tgt = full_mask_images[frame_idx][y1:y2 ,x1:x2, :]
                            bool_mask = mask.squeeze() > 0
                            tgt[bool_mask] = [255, 255, 255]

                    video_predictor.handle_request(request=dict(type="close_session", session_id=session_id))


            for frameId, image_path in enumerate(chunk_image_paths):
                if chunk.node.maskInvert.value:
                    mask = (full_mask_images[image_path[2]][:,:,0:1] == 0).astype('float32')
                else:
                    mask = (full_mask_images[image_path[2]][:,:,0:1] > 0).astype('float32')
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

                image.writeImage(outputFileMask, mask, sourceInfo["h_ori"], sourceInfo["w_ori"], sourceInfo["orientation"],
                                 sourceInfo["PAR"], metadata_deep_model, optWrite)

        finally:
            torch.cuda.empty_cache()


def get_image_paths_list(input_path, path_folder_x2 = "", path_folder_x4 = ""):
    from pyalicevision import sfmData, camera
    from pyalicevision import sfmDataIO
    from pathlib import Path

    image_paths = []

    if Path(input_path).suffix.lower() in [".sfm", ".abc"]:
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                for id, v in views.items():
                    image_x1_path = Path(v.getImage().getImagePath())
                    image_x1_name = image_x1_path.name
                    image_x2_path = None
                    if os.path.isfile(os.path.join(path_folder_x2, image_x1_name)):
                        image_x2_path = os.path.join(path_folder_x2, image_x1_name)
                    image_x4_path = None
                    if os.path.isfile(os.path.join(path_folder_x4, image_x1_name)):
                        image_x4_path = os.path.join(path_folder_x4, image_x1_name)
                    intrinsic = dataAV.getIntrinsicSharedPtr(v.getIntrinsicId())
                    pinhole = camera.Pinhole.cast(intrinsic)
                    par = 1.0
                    if pinhole is not None:
                        par = pinhole.getPixelAspectRatio()
                    image_paths.append((image_x1_path, str(id), v.getFrameId(), v.getImage().getWidth(),
                                        v.getImage().getHeight(), par, image_x2_path, image_x4_path))

            image_paths.sort(key=lambda x: x[0])
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path (folder or sfmData file).")
    return image_paths