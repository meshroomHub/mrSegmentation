__version__ = "1.0"

from chunk import Chunk
import chunk
import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
from pyalicevision import parallelization as avpar

import logging
logger = logging.getLogger("ViTMatte")

class ViTMatte(desc.Node):
    size = avpar.DynamicViewsSize("input")
    gpu = desc.Level.INTENSIVE
    parallelization = desc.Parallelization(blockSize=50)

    category = "Matting"
    documentation = """
Based on the SDMatte matting network, the node generates a matte from a binary mask.
The input binary mask is converted to a trimap before feeding the matting model
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
            description="Gaussian blur kernel size or Erode/Dilate rectangle kernel size",
            value=31,
            range=(3,1000,1),
        ),
        desc.ChoiceParam(
            name="inferenceSize",
            label="Inference Size",
            description="Image size for inference",
            value=512,
            values=[512, 640, 768, 896, 1024, 2048],
            exclusive=True,
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
            enabled=lambda node: node.inputMask.isLink,
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
                            if pathMask != "":
                                inputFileMask = os.path.join(pathMask, Path(inputFile).stem + "." + extMask)
                            outputFileMatte = os.path.join(outDir, Path(inputFile).stem + "." + extOut)
                            outputFileTrimap = os.path.join(outDir, "trimap_" + Path(inputFile).stem + "." + extOut)
                        else:
                            if pathMask != "":
                                inputFileMask = os.path.join(pathMask, str(id) + "." + extMask)
                            outputFileMatte = os.path.join(outDir, str(id) + "." + extOut)
                            outputFileTrimap = os.path.join(outDir, "trimap_" + str(id) + "." + extOut)
                        paths[inputFile] = (outputFileMatte, frameId, str(id), inputFileMask, outputFileTrimap)

        return paths

    def build_ViTMatte_model(self, device):

        if device not in ["cuda", "cpu"]:
            return None

        from detectron2.config import LazyConfig, instantiate
        from detectron2.checkpoint import DetectionCheckpointer

        config = os.getenv("VITMATTE_CONFIG_PATH") + '/common/model.py'
        cfg = LazyConfig.load(config)
        cfg.model.backbone.embed_dim = 768
        cfg.model.backbone.num_heads = 12
        cfg.model.decoder.in_chans = 768
        model = instantiate(cfg.model)
        model.to(device)
        model.eval()
        checkpoint = os.getenv("VITMATTE_MODELS_PATH") + "/ViTMatte_B_Com.pth"
        DetectionCheckpointer(model).load(checkpoint)

        return model


    def processChunk(self, chunk):
        from segmentationRDS import image

        from torchvision.transforms import functional as F
        import cv2

        import numpy as np
        import torch
        from pyalicevision import image as avimg

        try:
            logger.setLevel(chunk.node.verboseLevel.value.upper())

            if not chunk.node.input:
                logger.warning("Nothing to segment")
                return
            if not chunk.node.output.value:
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

            outFiles = self.resolvedPaths(chunk.node.input.value,
                                          pathMask, chunk.node.extensionMask.value,
                                          chunk.node.output.value, chunk.node.keepFilename.value,
                                          chunk.node.extensionOut.value)

            if promptType == "":
                raise ValueError("Some images have no valid mask or trimap to drive the matting process !!!")
            else:
                logger.info(f"prompt type: {promptType}")

                if not os.path.exists(chunk.node.output.value):
                    os.mkdir(chunk.node.output.value)

                device = "cuda" if torch.cuda.is_available() and chunk.node.useGpu.value else "cpu"

                model = self.build_ViTMatte_model(device)

                if model is None:
                    raise ValueError("SDMatte model cannot be loaded")

                metadata_deep_model = {}
                metadata_deep_model["Meshroom:mrSegmentation:DeepModelName"] = "ViTMatte"
                metadata_deep_model["Meshroom:mrSegmentation:DeepModelVersion"] = "B_Com"

                sz = int(chunk.node.inferenceSize.value)

                for k, (iFile, oFile) in enumerate(outFiles.items()):
                    if k >= chunk.range.start and k <= chunk.range.last:

                        img, h_ori, w_ori, PAR, orientation = image.loadImage(iFile, True)
                        frameId = oFile[1]
                        viewId = oFile[2]

                        logger.info("frameId: {} - {}".format(frameId, iFile))

                        if w_ori > h_ori and w_ori > sz:
                            inference_size = (sz, int(sz * h_ori / w_ori))
                        elif h_ori > w_ori and h_ori > sz:
                            inference_size = (int(sz * w_ori / h_ori), sz)
                        elif w_ori == h_ori and w_ori > sz:
                            inference_size = (sz, sz)
                        img_sized = cv2.resize(img, inference_size, interpolation=cv2.INTER_LINEAR)
                        img_tensor = F.to_tensor(img_sized).unsqueeze(0)

                        sample = {"image": img_tensor}

                        promptRGB, h_ori_mask, w_ori_mask, PAR_mask, orientation_mask = image.loadImage(oFile[3], True)
                        promptRGB_sized = cv2.resize(promptRGB, inference_size, interpolation=cv2.INTER_NEAREST)
                        if promptType == "trimap":
                            trimap = promptRGB_sized[:, :, 0]
                        else:
                            kernel_size = chunk.node.kernelSize.value
                            trimap = np.full(promptRGB_sized.shape, 0.5, dtype=np.float32)
                            if chunk.node.trimapComputationMethod.value == "Blur":
                                blurred_mask = cv2.GaussianBlur(promptRGB_sized, (kernel_size, kernel_size), 0)
                                threshold_low = 0.05
                                threshold_high = 0.95
                                trimap[blurred_mask >= threshold_high] = 1.0
                                trimap[blurred_mask <= threshold_low] = 0.0
                            else:
                                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                                mask_dilate = cv2.dilate(promptRGB_sized, kernel, iterations=1)
                                mask_erode = cv2.erode(promptRGB_sized, kernel, iterations=1)
                                trimap[mask_erode > 0.0] = 1.0
                                trimap[mask_dilate == 0.0] = 0.0

                        sample["trimap"] = F.to_tensor(trimap[:, :, 0]).unsqueeze(0)

                        with torch.no_grad():
                            matte = model(sample)['phas'].flatten(0, 2)
                            matte = cv2.resize(matte.detach().cpu().numpy(), (w_ori, h_ori))
                            # pred = model(sample)
                            # matte = pred.flatten(0, 2)
                            matteRGB = np.dstack([matte, matte, matte])

                        optWrite = avimg.ImageWriteOptions()
                        if Path(oFile[0]).suffix.lower() == ".exr":
                            optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
                            optWrite.exrCompressionMethod(avimg.EImageExrCompression_stringToEnum("DWAA"))
                            optWrite.exrCompressionLevel(300)
                        else:
                            optWrite.toColorSpace(avimg.EImageColorSpace_SRGB)

                        image.writeImage(oFile[0], matteRGB, h_ori, w_ori, orientation, PAR, metadata_deep_model, optWrite)
                        if promptType == "mask":
                            image.writeImage(oFile[4], trimap, h_ori, w_ori, orientation, PAR, metadata_deep_model, optWrite)

        finally:
            torch.cuda.empty_cache()
