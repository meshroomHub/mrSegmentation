__version__ = "0.1"

from chunk import Chunk
import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

import logging
logger = logging.getLogger("SDMatte")

class SDMatteNodeSize(desc.MultiDynamicNodeSize):
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
        
class SDMatte(desc.Node):
    size = SDMatteNodeSize(['input', 'extensionIn'])
    gpu = desc.Level.INTENSIVE
    parallelization = desc.Parallelization(blockSize=50)

    category = "Matting"
    documentation = """
Based on the SDMatte matting network, the node generates a matte from a bounding box, a mask or a set of points.
Only one type of driving prompt must be provided for a given image.
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
            group="",  # remove from command line params
        ),
        desc.ChoiceParam(
            name="inferenceSize",
            label="Inference Size",
            description="Image size for inference",
            value=512,
            values=[512, 640, 768, 896, 1024],
            exclusive=True,
        ),
        desc.StringParam(
            name="caption",
            label="Caption",
            description="Caption",
            value="",
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
        desc.ShapeList(
            name="positiveClicks",
            label="Positive Clicks",
            description="Clicks driving the matting operations",
            shape=desc.Point2d(
                name="click",
                label="Click",
                description="Point.",
                keyable=True,
                keyType="viewId",
            ),
        ),
        desc.ShapeList(
            name="positiveBoxes",
            label="Positive Boxes",
            description="Bounding Boxes acting as RoIs for the matting operations.",
            shape=desc.Rectangle(
                name="bbox",
                label="Bounding Box",
                description="Rectangle.",
                keyable=True,
                keyType="viewId",
            ),
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
    ]

    def resolvedPaths(self, pathIn, extIn, pathMask, extMask, outDir, keepFilename, extOut):
        from pyalicevision import sfmData
        from pyalicevision import sfmDataIO
        from pathlib import Path
        import itertools

        include_suffixes = [extIn.lower(), extIn.upper()]
        paths = {}
        inputFileMask = None
        if Path(pathIn).is_dir():
            input_filepaths = sorted(itertools.chain(*(Path(pathIn).glob(f'*.{suffix}') for suffix in include_suffixes)))
            for frameId, inputFile in enumerate(input_filepaths):
                if pathMask != "":
                    inputFileMask = os.path.join(pathMask, Path(inputFile).stem + "." + extMask)
                outputFileMatte = os.path.join(outDir, Path(inputFile).stem + "." + extOut)
                paths[str(inputFile)] = (outputFileMatte, frameId, 'not_a_view', inputFileMask)
        elif Path(pathIn).suffix.lower() in [".sfm", ".abc"]:
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
                        else:
                            if pathMask != "":
                                inputFileMask = os.path.join(pathMask, str(id) + "." + extMask)
                            outputFileMatte = os.path.join(outDir, str(id) + "." + extOut)
                        paths[inputFile] = (outputFileMatte, frameId, str(id), inputFileMask)

        return paths

    def getClickDictWithViewIdAsKeyFromShape(self, shapeList):
        clickDictFromShape = {}
        shapes = shapeList.getShapesAsDict()
        if shapes:
            for sh in shapes:
                color = sh["properties"]["color"]
                for key in sh["observations"]:
                    x = sh["observations"][key]["x"]
                    y = sh["observations"][key]["y"]
                    pt = [x, y]
                    if key in clickDictFromShape:
                        clickDictFromShape[key].append((pt, color))
                    else:
                        clickDictFromShape[key] = [(pt, color)]
        return clickDictFromShape

    def normalize_click(self, click_xy, img_w, img_h, PAR, orientation):
        from segmentationRDS import image
        normalized_click = click_xy.copy()
        normalized_click[0], normalized_click[1] = image.fromRawToUsualOrientation(normalized_click[0], normalized_click[1], img_w, img_h, PAR, orientation)
        normalized_click[0] /= img_w
        normalized_click[1] /= img_h
        return normalized_click

    def get_clicks_xy(self, clicks_xy, img_w, img_h, PAR, orientation):
        clicks_xy_norm = []
        for click_xy in clicks_xy:
            clicks_xy_norm.append(self.normalize_click(click_xy, img_w, img_h, PAR, orientation))
        clicks_xy_out = []
        clicks_xy_norm_out = []
        for click_xy_norm in clicks_xy_norm:
            x_norm = min(max(click_xy_norm[0], 0), 1)
            y_norm = min(max(click_xy_norm[1], 0), 1)
            clicks_xy_norm_out.append([x_norm, y_norm])
            clicks_xy_out.append([int(x_norm * img_w), int(y_norm * img_h)])
        return clicks_xy_out, clicks_xy_norm_out

    def getBboxDictWithViewIdAsKeyFromShape(self, shape):
        bboxDictFromShape = {}
        shapesBBoxesIn = shape.getShapesAsDict()
        if shapesBBoxesIn:
            for sh in shapesBBoxesIn:
                for key in sh["observations"]:
                    xc = sh["observations"][key]["center"]["x"]
                    yc = sh["observations"][key]["center"]["y"]
                    w = sh["observations"][key]["size"]["width"]
                    h = sh["observations"][key]["size"]["height"]
                    bb = [xc - w/2, yc - h/2, w, h]
                    if key in bboxDictFromShape:
                        bboxDictFromShape[key].append(bb)
                    else:
                        bboxDictFromShape[key] = [bb]
        return bboxDictFromShape

    def normalize_bbox(self, bbox_xywh, img_w, img_h, PAR, orientation):
        import torch
        from segmentationRDS import image
        # Assumes bbox_xywh is in XYWH format
        if isinstance(bbox_xywh, list):
            assert (
                len(bbox_xywh) == 4
            ), "bbox_xywh list must have 4 elements. Batching not supported except for torch tensors."
            normalized_bbox = bbox_xywh.copy()
            normalized_bbox[0], normalized_bbox[1] = image.fromRawToUsualOrientation(normalized_bbox[0], normalized_bbox[1], img_w, img_h, PAR, orientation)
            normalized_bbox[2], normalized_bbox[3] = image.fromRawToUsualOrientation(normalized_bbox[2], normalized_bbox[3], img_w, img_h, PAR, orientation)
            normalized_bbox[0] /= img_w
            normalized_bbox[1] /= img_h
            normalized_bbox[2] /= img_w
            normalized_bbox[3] /= img_h
        else:
            assert isinstance(
                bbox_xywh, torch.Tensor
            ), "Only torch tensors are supported for batching."
            normalized_bbox = bbox_xywh.clone()
            assert (
                normalized_bbox.size(-1) == 4
            ), "bbox_xywh tensor must have last dimension of size 4."
            normalized_bbox[..., 0], normalized_bbox[..., 1] = image.fromRawToUsualOrientation(normalized_bbox[..., 0], normalized_bbox[..., 1], img_w, img_h, PAR, orientation)
            normalized_bbox[..., 2], normalized_bbox[..., 3] = image.fromRawToUsualOrientation(normalized_bbox[..., 2], normalized_bbox[..., 3], img_w, img_h, PAR, orientation)
            normalized_bbox[..., 0] /= img_w
            normalized_bbox[..., 1] /= img_h
            normalized_bbox[..., 2] /= img_w
            normalized_bbox[..., 3] /= img_h
        return normalized_bbox

    def get_bboxes_xyxy(self, bboxes_xywh, img_w, img_h, PAR, orientation):
        bboxes_xywh_norm = []
        for bbox_xywh in bboxes_xywh:
            bboxes_xywh_norm.append(self.normalize_bbox(bbox_xywh, img_w, img_h, PAR, orientation))
        bboxes_xyxy = []
        bboxes_xyxy_norm = []
        for bbox_xywh_norm in bboxes_xywh_norm:
            x1_norm = min(max(bbox_xywh_norm[0], 0), 1)
            y1_norm = min(max(bbox_xywh_norm[1], 0), 1)
            x2_norm = min(max(bbox_xywh_norm[0] + bbox_xywh_norm[2], 0), 1)
            y2_norm = min(max(bbox_xywh_norm[1] + bbox_xywh_norm[3], 0), 1)
            bboxes_xyxy_norm.append([x1_norm, y1_norm, x2_norm, y2_norm])
            bboxes_xyxy.append([int(x1_norm * img_w), int(y1_norm * img_h), int(x2_norm * img_w), int(y2_norm * img_h)])
        return bboxes_xyxy, bboxes_xyxy_norm

    def build_SDMatte_model(self, modelFolder, checkpoint, device, promptType):

        if device not in ["cuda", "cpu"] or promptType not in ["bbox_mask", "point_mask", "mask", "trimap"]:
            return None

        import torch
        from modeling import SDMatte
        from detectron2.checkpoint import DetectionCheckpointer
        from packaging import version

        if version.parse(torch.__version__) > version.Version("2.5"):
            import omegaconf
            import typing
            import collections
            torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])
            torch.serialization.add_safe_globals([omegaconf.base.ContainerMetadata])
            torch.serialization.add_safe_globals([typing.Any])
            torch.serialization.add_safe_globals([list])
            torch.serialization.add_safe_globals([collections.defaultdict])
            torch.serialization.add_safe_globals([dict])
            torch.serialization.add_safe_globals([int])
            torch.serialization.add_safe_globals([omegaconf.nodes.AnyNode])
            torch.serialization.add_safe_globals([omegaconf.base.Metadata])

        torch.set_grad_enabled(False)

        model = SDMatte(
            pretrained_model_name_or_path=modelFolder,
            load_weight=False,
            conv_scale=3,
            num_inference_steps=1,
            aux_input=promptType,
            add_noise=False,
            use_dis_loss=True,
            use_aux_input=True,
            use_coor_input=True,
            use_attention_mask=True,
            residual_connection=False,
            use_encoder_hidden_states=True,
            use_attention_mask_list=[True, True, True],
            use_encoder_hidden_states_list=[False, True, False],
        )
        model.to(device)
        DetectionCheckpointer(model).load(checkpoint)
        model.eval()

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

            outFiles = self.resolvedPaths(chunk.node.input.value, chunk.node.extensionIn.value,
                                          pathMask, chunk.node.extensionMask.value,
                                          chunk.node.output.value, chunk.node.keepFilename.value,
                                          chunk.node.extensionOut.value)

            bboxOK = False
            clicksOK = False
            if pathMask == "":
                posBboxDictFromShape = self.getBboxDictWithViewIdAsKeyFromShape(chunk.node.positiveBoxes)
                bboxOK = True
                for k, (iFile, oFile) in enumerate(outFiles.items()):
                    viewId = oFile[2]
                    if viewId not in posBboxDictFromShape:
                        bboxOK = False
                        break
                if not bboxOK:
                    posClickDictFromShape = self.getClickDictWithViewIdAsKeyFromShape(chunk.node.positiveClicks)
                    clicksOK = True
                    for k, (iFile, oFile) in enumerate(outFiles.items()):
                        viewId = oFile[2]
                        if viewId not in posClickDictFromShape:
                            clicksOK = False
                            break

            if promptType == "" and bboxOK:
                promptType = "bbox_mask"
            elif promptType == "" and clicksOK:
                promptType = "point_mask"
            elif promptType == "" and chunk.node.caption.value != "":
                promptType = "auto_mask"
                logger.warning("No graphical prompt provided, caption will be combined with a full mask.")

            if promptType == "":
                raise ValueError("Some images have no valid prompt to drive the matting process !!!")
            else:

                if not os.path.exists(chunk.node.output.value):
                    os.mkdir(chunk.node.output.value)

                device = "cuda" if torch.cuda.is_available() and chunk.node.useGpu.value else "cpu"

                modelFolder = os.getenv("SDMATTE_MODELS_PATH")
                if modelFolder is None:
                    raise ValueError("SDMATTE_MODELS_PATH env var is not defined.")

                modelCheckpoint = os.getenv("SDMATTE_MODELS_PATH") + "/SDMatte.pth"

                model = self.build_SDMatte_model(modelFolder, modelCheckpoint, device, promptType if promptType != "auto_mask" else "mask")

                if model is None:
                    raise ValueError("SDMatte model cannot be loaded")

                metadata_deep_model = {}
                metadata_deep_model["Meshroom:mrSegmentation:DeepModelName"] = "SDMatte"
                metadata_deep_model["Meshroom:mrSegmentation:DeepModelVersion"] = "0.1"

                for k, (iFile, oFile) in enumerate(outFiles.items()):
                    if k >= chunk.range.start and k <= chunk.range.last:

                        img, h_ori, w_ori, PAR, orientation = image.loadImage(iFile, True)
                        frameId = oFile[1]
                        viewId = oFile[2]

                        logger.info("frameId: {} - {}".format(frameId, iFile))

                        sample = {}
                        sz = int(chunk.node.inferenceSize.value)
                        inference_size = (sz, sz)
                        img_sized = cv2.resize(img, inference_size, interpolation=cv2.INTER_LINEAR)
                        img_scaled = img_sized.copy() * 2 - 1
                        sample["image"] = F.to_tensor(img_scaled).float().unsqueeze(0)
                        sample["is_trans"] = torch.tensor(0).long().unsqueeze(0)
                        sample["caption"] = chunk.node.caption.value

                        if promptType in ["mask", "trimap"]:
                            maskRGB, h_ori_mask, w_ori_mask, PAR_mask, orientation_mask = image.loadImage(oFile[3], True)
                            mask = maskRGB[:,:,0]
                            mask_sized = cv2.resize(mask, inference_size, interpolation=cv2.INTER_NEAREST)
                            mask_scaled = mask_sized.copy() * 2 - 1
                            sample["mask"] = F.to_tensor(mask_scaled).float().unsqueeze(0)
                            sample["mask_coords"] = np.array([0, 0, 1, 1])
                            sample["mask_coords"] = torch.from_numpy(sample["mask_coords"]).float().unsqueeze(0)
                        elif promptType == "auto_mask":
                            mask = np.ones_like(img)[:,:,0]
                            mask_sized = cv2.resize(mask, inference_size, interpolation=cv2.INTER_NEAREST)
                            mask_scaled = mask_sized.copy() * 2 - 1
                            sample["mask"] = F.to_tensor(mask_scaled).float().unsqueeze(0)
                            sample["mask_coords"] = np.array([0, 0, 1, 1])
                            sample["mask_coords"] = torch.from_numpy(sample["mask_coords"]).float().unsqueeze(0)
                        elif promptType == "bbox_mask":
                            bboxes_xywh = posBboxDictFromShape[viewId]
                            bboxes_xyxy, bboxes_xyxy_norm = self.get_bboxes_xyxy(bboxes_xywh, img.shape[1], img.shape[0], PAR, orientation)
                            logger.debug(f"bboxes_xyxy = {bboxes_xyxy}")
                            logger.debug(f"bboxes_xyxy_norm = {bboxes_xyxy_norm}")
                            bbox_mask = np.zeros_like(img)[:,:,0]
                            bbox_mask[bboxes_xyxy[0][1]:bboxes_xyxy[0][3], bboxes_xyxy[0][0]:bboxes_xyxy[0][2]] = 1
                            bbox_mask_sized = cv2.resize(bbox_mask, inference_size, interpolation=cv2.INTER_NEAREST)
                            bbox_mask_scaled = bbox_mask_sized.copy() * 2 - 1
                            sample["bbox_mask"] = F.to_tensor(bbox_mask_scaled).float().unsqueeze(0)
                            sample["bbox_coords"] = np.array(bboxes_xyxy_norm[0])
                            sample["bbox_coords"] = torch.from_numpy(sample["bbox_coords"]).float().unsqueeze(0)
                        elif promptType == "point_mask":
                            import scipy
                            clicks = [pt[0] for pt in posClickDictFromShape[viewId]]
                            clicks_xy, clicks_xy_norm = self.get_clicks_xy(clicks, img.shape[1], img.shape[0], PAR, orientation)
                            logger.debug(f"clicks_xy = {clicks_xy}")
                            logger.debug(f"clicks_xy_norm = {clicks_xy_norm}")
                            point_mask = np.zeros_like(img)[:,:,0]
                            point_coords = []
                            for idx, click_xy in enumerate(clicks_xy):
                                tmp_mask = np.zeros_like(img)[:,:,0]
                                tmp_mask[click_xy[1], click_xy[0]] = 1
                                tmp_mask = scipy.ndimage.gaussian_filter(tmp_mask, sigma=20)
                                tmp_mask /= np.max(tmp_mask)
                                point_mask = np.maximum(point_mask, tmp_mask)
                                point_coords.append(clicks_xy_norm[idx][0])
                                point_coords.append(clicks_xy_norm[idx][1])
                            point_mask_sized = cv2.resize(point_mask, inference_size, interpolation=cv2.INTER_NEAREST)
                            point_mask_scaled = point_mask_sized.copy() * 2 - 1
                            sample["point_mask"] = F.to_tensor(point_mask_scaled).float().unsqueeze(0)
                            sample["point_coords"] = np.array(point_coords)
                            sample["point_coords"] = torch.from_numpy(sample["point_coords"]).float().unsqueeze(0)

                        with torch.no_grad():
                            pred = model(sample)
                            matte = pred.flatten(0, 2)
                            matte = cv2.resize(matte.detach().cpu().numpy(), (w_ori, h_ori))
                            matteRGB = np.dstack([matte, matte, matte])

                        optWrite = avimg.ImageWriteOptions()
                        if Path(oFile[0]).suffix.lower() == ".exr":
                            optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
                            optWrite.exrCompressionMethod(avimg.EImageExrCompression_stringToEnum("DWAA"))
                            optWrite.exrCompressionLevel(300)
                        else:
                            optWrite.toColorSpace(avimg.EImageColorSpace_SRGB)

                        image.writeImage(oFile[0], matteRGB, h_ori, w_ori, orientation, PAR, metadata_deep_model, optWrite)

        finally:
            torch.cuda.empty_cache()
