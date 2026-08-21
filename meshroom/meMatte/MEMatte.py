__version__ = "1.0"

import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
from pyalicevision import parallelization as avpar

import logging
logger = logging.getLogger("MEMatte")

class MEMatte(desc.Node):
    size = avpar.DynamicViewsSize("input")
    gpu = desc.Level.INTENSIVE
    parallelization = desc.Parallelization(blockSize=50)

    category = "Matting"
    documentation = """
This module implements a Meshroom matting node based on the MEMatte 
(Memory Efficient Matting) network. It generates high-quality alpha
mattes from images using either binary masks or trimaps as input prompts.

The general processing pipeline is:
-----------------------------------
- 1. Load SfM data and associated masks/trimaps
- 2. If a binary mask is provided, convert it to a trimap via Gaussian blur or morphological operations
- 3. For each detected object (bounding box), crop the region of interest
- 4. Resize the region if it exceeds the maximum inference size
- 5. Run MEMatte inference to generate the alpha matte
- 6. Save the output mattes and trimaps

Available Models:
-----------------
- Large-Comp1k: MEMatte_ViTB_DIM ; Trained on Composition-1k
- Small-Comp1k: MEMatte_ViTS_DIM ; Lightweight version
- Small-AIM500: MEMatte          ; Trained on AIM500

Known Limitations:
------------------
- Requires a SfMData file (.sfm or .abc) as input.
- Either a mask or a trimap must be provided, but not both simultaneously.
- If no trimap transition region (value 0.5) is found in a bounding box,
  processing stops with an error for that frame.

"""

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
            enabled=lambda node: node.inputTrimap.value == "",
        ),
        desc.File(
            name="inputTrimap",
            label="Trimap Folder",
            description="Folder containing the trimaps used as prompt.",
            value="",
            enabled=lambda node: node.inputMask.value == "",
        ),
        desc.ChoiceParam(
            name="extensionMask",
            label="Mask/Trimap File Extension",
            description="Input mask or trimap file extension.",
            value="exr",
            values=["exr", "png", "jpg"],
            exclusive=True,
        ),
        desc.ChoiceParam(
            name="trimapComputationMethod",
            label="Trimap Computation Method",
            description="Method to compute a trimap from a binary mask.",
            value="Blur",
            values=["Blur", "Morpho"],
            exclusive=True,
        ),
        desc.IntParam(
            name="kernelSize",
            label="Kernel Size",
            description="Gaussian blur kernel size or Erode/Dilate rectangle kernel size.",
            value=31,
            range=(3,1000,1),
        ),
        desc.ChoiceParam(
            name="meMatteModel",
            label="MEMatte Model",
            description="Model for inference.",
            value="Large-Comp1k",
            values=["Large-Comp1k", "Small-Comp1k", "Small-AIM500"],
            exclusive=True,
        ),
        desc.IntParam(
            name="maxTokenNumber",
            label="Maximum Token Number",
            description="Maximum number of tokens use for inference.",
            value=18500,
            range=(10000, 20000, 100),
        ),
        desc.BoolParam(
            name="limitInferenceSize",
            label="Limit Inference Size",
            description="If enabled, input image and trimap will be resized if needed.",
            value=True,
        ),
        desc.ChoiceParam(
            name="inferenceSize",
            label="Inference Size Max",
            description="Maximum size of the largest image dimension for inference. Automatic resize if higher.",
            value=1024,
            values=[512, 640, 768, 896, 1024, 2048],
            exclusive=True,
            enabled=lambda node: node.limitInferenceSize.value,
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
        desc.File(
            name="trimap",
            label="Trimap",
            description="Trimap computed from input mask.",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/trimap_" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + "." + attr.node.extensionOut.value,
            enabled=lambda node: node.inputMask.value != "",
        ),
    ]

    def resolvedPaths(self, pathIn, pathMask, extMask, outDir, keepFilename, extOut):
        from pyalicevision import sfmData
        from pyalicevision import sfmDataIO
        from pathlib import Path

        paths = {}
        inputFileMask = None
        if Path(pathIn).suffix.lower() in [".sfm", ".abc"]:
            if Path(pathIn).exists():
                dataAV = sfmData.SfMData()
                if sfmDataIO.load(dataAV, pathIn, sfmDataIO.ALL) and os.path.isdir(outDir):
                    views = dataAV.getViews()
                    for id, v in views.items():
                        inputFile = v.getImage().getImagePath()
                        frameId = v.getFrameId()
                        if keepFilename:
                            if pathMask:
                                inputFileMask = os.path.join(pathMask, Path(inputFile).stem + "." + extMask)
                            outputFileMatte = os.path.join(outDir, Path(inputFile).stem + "." + extOut)
                            outputFileTrimap = os.path.join(outDir, "trimap_" + Path(inputFile).stem + "." + extOut)
                        else:
                            if pathMask:
                                inputFileMask = os.path.join(pathMask, str(id) + "." + extMask)
                            outputFileMatte = os.path.join(outDir, str(id) + "." + extOut)
                            outputFileTrimap = os.path.join(outDir, "trimap_" + str(id) + "." + extOut)
                        paths[inputFile] = (outputFileMatte, frameId, str(id), inputFileMask, outputFileTrimap)

        return paths

    def build_MEMatte_model(self, configPath, checkpointPath, maxTokenNumber, device):

        if device not in ["cuda", "cpu"]:
            return None

        from detectron2.config import LazyConfig, instantiate
        from detectron2.checkpoint import DetectionCheckpointer

        cfg = LazyConfig.load(configPath)
        cfg.model.teacher_backbone = None
        cfg.model.backbone.max_number_token = maxTokenNumber
        model = instantiate(cfg.model)
        model.to(device)
        model.eval()
        DetectionCheckpointer(model).load(checkpointPath)

        return model


    def processChunk(self, chunk):
        from segmentationRDS import image, bboxUtils

        from torchvision.transforms import functional as F
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

            pathMask = ""
            promptType = ""
            if chunk.node.inputMask.value != "":
                pathMask = chunk.node.inputMask.value
                promptType = "mask"
            elif chunk.node.inputTrimap.value != "":
                pathMask = chunk.node.inputTrimap.value
                promptType = "trimap"
            else:
                raise ValueError("Some images have no valid mask or trimap to drive the matting process!")
            logger.info(f"prompt type: {promptType}")

            modelConfigNames = {"Large-Comp1k": "MEMatte_B_topk0.25_win_global_long.py",
                                "Small-Comp1k": "MEMatte_S_topk0.25_win_global_long.py",
                                "Small-AIM500": "MEMatte_S_topk0.25_win_global_long.py"}
            checkpointNames = {"Large-Comp1k": "MEMatte_ViTB_DIM.pth",
                               "Small-Comp1k": "MEMatte_ViTS_DIM.pth",
                               "Small-AIM500": "MEMatte.pth"}

            outFiles = self.resolvedPaths(chunk.node.input.value,
                                          pathMask, chunk.node.extensionMask.value,
                                          chunk.node.output.value, chunk.node.keepFilename.value,
                                          chunk.node.extensionOut.value)

            if not os.path.exists(chunk.node.output.value):
                os.mkdir(chunk.node.output.value)

            device = "cuda" if torch.cuda.is_available() and chunk.node.useGpu.value else "cpu"

            configRoot = os.getenv("MEMATTE_CONFIG_PATH")
            checkpointRoot = os.getenv("MEMATTE_MODELS_PATH")
            if not configRoot or not checkpointRoot:
                raise ValueError("MEMATTE_CONFIG_PATH and MEMATTE_MODELS_PATH must both be defined.")
            configPath = configRoot + "/" + modelConfigNames[chunk.node.meMatteModel.value]
            checkpointPath = checkpointRoot + "/" + checkpointNames[chunk.node.meMatteModel.value]

            logger.info(f"configPath = {configPath}")
            logger.info(f"checkpointPath = {checkpointPath}")

            model = self.build_MEMatte_model(configPath, checkpointPath, chunk.node.maxTokenNumber.value, device)

            if model is None:
                raise ValueError("MEMatte model cannot be loaded")

            metadata_deep_model_base = {
            "Meshroom:mrSegmentation:DeepModelName": "MEMatte",
            "Meshroom:mrSegmentation:DeepModelVersion": chunk.node.meMatteModel.value + "-" + str(chunk.node.maxTokenNumber.value)
            }

            limitedSize = chunk.node.limitInferenceSize.value
            sz = int(chunk.node.inferenceSize.value)

            for k, (iFile, oFile) in enumerate(outFiles.items()):
                if k >= chunk.range.start and k <= chunk.range.last:

                    img, h_ori, w_ori, PAR, orientation = image.loadImage(iFile, True)
                    frameId = oFile[1]
                    viewId = oFile[2]
                    oiioImgBuf = oiio.ImageBuf(oFile[3])
                    metadata_deep_model = dict(metadata_deep_model_base)
                    metadata = oiioImgBuf.spec().extra_attribs
                    boxes = {}
                    for m in metadata:
                        if not m.name.startswith("Meshroom:mrSegmentation:"):
                            continue
                        underscore_pos = m.name.rfind("_")
                        if underscore_pos == -1:
                            continue
                        suffix = m.name[underscore_pos + 1:]
                        if len(suffix) >= 1 and suffix.isdigit():
                            boxes[m.name] = [int(v) for v in m.value.split(';')]
                    logger.info("frameId: {} - {}".format(frameId, iFile))
                    logger.info(f"boxes: {boxes}")

                    promptRGB, h_ori_mask, w_ori_mask, PAR_mask, orientation_mask = image.loadImage(oFile[3], True)

                    matteRGB = np.zeros_like(img)
                    fullTrimap = np.zeros_like(img)

                    for key, box in boxes.items():
                        x1, y1, x2, y2 = bboxUtils.box_to_display(box, PAR)
                        mask_box = np.zeros_like(promptRGB)
                        mask_box[y1:y2, x1:x2, :] = promptRGB[y1:y2, x1:x2, :]
                        if promptType == "trimap":
                            trimap_box = mask_box[:, :, 0]
                        else:
                            kernel_size = chunk.node.kernelSize.value
                            trimap_box = np.full(promptRGB.shape, 0.5, dtype=np.float32)
                            if chunk.node.trimapComputationMethod.value == "Blur":
                                if kernel_size % 2 == 0:
                                    kernel_size += 1
                                blurred_mask = cv2.GaussianBlur(mask_box, (kernel_size, kernel_size), 0)
                                threshold_low = 0.05
                                threshold_high = 0.95
                                trimap_box[blurred_mask >= threshold_high] = 1.0
                                trimap_box[blurred_mask <= threshold_low] = 0.0
                            else:
                                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                                mask_dilate = cv2.dilate(mask_box, kernel, iterations=1)
                                mask_erode = cv2.erode(mask_box, kernel, iterations=1)
                                trimap_box[mask_erode > 0.0] = 1.0
                                trimap_box[mask_dilate == 0.0] = 0.0
                            trimap_box = trimap_box[:, :, 0]

                        coords = np.argwhere(np.isclose(trimap_box, 0.5, atol=1.0 / 255.0))
                        if len(coords) == 0:
                            logger.warning(f"No trimap in box {(x1, y1, x2, y2)} af frame {frameId}")
                            return

                        ys = coords[:, 0]
                        xs = coords[:, 1]
                        x_left = xs.min()
                        x_right = xs.max() + 1
                        y_top = ys.min()
                        y_bottom = ys.max() + 1
                        x_center = (x_right + x_left) // 2
                        y_center = (y_top + y_bottom) // 2
                        width = x_right - x_left
                        height = y_bottom - y_top

                        trimap_ext = 1.1
                        mask_height, mask_width, _ = promptRGB.shape
                        x_half_extent = max(1, int(trimap_ext * width / 2.0))
                        y_half_extent = max(1, int(trimap_ext * height / 2.0))
                        x_left_inference = max(0, x_center - x_half_extent)
                        x_right_inference = min(mask_width, x_center + x_half_extent)
                        y_top_inference = max(0, y_center - y_half_extent)
                        y_bottom_inference = min(mask_height, y_center + y_half_extent)
                        top_left_xy = (x_left_inference, y_top_inference)
                        bottom_right_xy = (x_right_inference, y_bottom_inference)

                        w_inference = x_right_inference - x_left_inference
                        h_inference = y_bottom_inference - y_top_inference

                        logger.debug(f"box for inference : [{top_left_xy}, {bottom_right_xy}] ; ({w_inference} x {h_inference})")

                        img_for_inference = img[y_top_inference:y_bottom_inference, x_left_inference:x_right_inference, :]
                        trimap_for_inference = trimap_box[y_top_inference:y_bottom_inference, x_left_inference:x_right_inference]

                        if limitedSize:
                            if w_inference > h_inference and w_inference > sz:
                                inference_size = (sz, int(sz * h_inference / w_inference))
                            elif h_inference > w_inference and h_inference > sz:
                                inference_size = (int(sz * w_inference / h_inference), sz)
                            elif w_inference == h_inference and w_inference > sz:
                                inference_size = (sz, sz)
                            else:
                                inference_size = (w_inference, h_inference)
                            logger.debug(f"inference size : {inference_size} ; ratio = {inference_size[0] / w_inference}")

                            img_sized = cv2.resize(img_for_inference, inference_size, interpolation=cv2.INTER_LINEAR)
                            box_trimap = np.expand_dims(cv2.resize(trimap_for_inference, inference_size, interpolation=cv2.INTER_NEAREST), axis=-1)
                        else:
                            img_sized = img_for_inference
                            box_trimap = np.expand_dims(trimap_for_inference, axis=-1)

                        sample = {"image": torch.from_numpy(img_sized).permute(2, 0, 1).unsqueeze(0)}
                        sample["trimap"] = torch.from_numpy(box_trimap).permute(2, 0, 1).unsqueeze(0)

                        with torch.no_grad():
                            matte, _, _ = model(sample, patch_decoder=True)
                            matte = matte['phas'].flatten(0, 2).squeeze(0)
                            matte = matte.detach().cpu().numpy()
                            if limitedSize:
                                matte = cv2.resize(matte, (w_inference, h_inference), interpolation=cv2.INTER_LINEAR)
                            matte[trimap_for_inference == 0] = 0
                            matte[trimap_for_inference == 1] = 1
                            box_matteRGB = np.dstack([matte, matte, matte])
                            tgt = matteRGB[y_top_inference:y_bottom_inference, x_left_inference:x_right_inference, :]
                            matteRGB[y_top_inference:y_bottom_inference, x_left_inference:x_right_inference, :] = np.maximum(tgt, box_matteRGB)

                        trimap_for_inference = np.dstack([trimap_for_inference, trimap_for_inference, trimap_for_inference])
                        tgt = fullTrimap[y_top_inference:y_bottom_inference, x_left_inference:x_right_inference, :]
                        fullTrimap[y_top_inference:y_bottom_inference, x_left_inference:x_right_inference, :] = np.maximum(tgt, trimap_for_inference)

                        metadata_deep_model[key] = str(x_left_inference) + ";" + str(y_top_inference) + ";"
                        metadata_deep_model[key] += str(x_right_inference) + ";" + str(y_bottom_inference)

                    optWrite = avimg.ImageWriteOptions()
                    if Path(oFile[0]).suffix.lower() == ".exr":
                        optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
                        optWrite.exrCompressionMethod(avimg.EImageExrCompression_stringToEnum("DWAA"))
                        optWrite.exrCompressionLevel(300)
                    else:
                        optWrite.toColorSpace(avimg.EImageColorSpace_SRGB)

                    image.writeImage(oFile[0], matteRGB, h_ori, w_ori, orientation, PAR, metadata_deep_model, optWrite)
                    if promptType == "mask":
                        image.writeImage(oFile[4], fullTrimap, h_ori, w_ori, orientation, PAR, metadata_deep_model, optWrite)

        finally:
            torch.cuda.empty_cache()
