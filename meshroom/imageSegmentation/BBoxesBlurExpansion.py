__version__ = "1.0"

import json
import os

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL


class BBoxesBlurExpansion(desc.Node):

    category = "Utils"
    documentation = """
Expands the bounding boxes of a bboxes.json file by a margin in pixels, so that the semi-transparent parts of blurred
objects are not cut by the crop of the matting nodes (VideoMatting...).

For every box: margin = Min Box Margin + Blur Margin Factor x percentile(motion blur length inside the box).
The motion blur length maps are the "_motionBlurLength" outputs of the ImageBlurMap node.
Without blur maps, only the minimum margin is applied.

The output file has the same structure as the input one (all labels and tracking keys are processed), but its boxes
are no longer tight boxes around the masks: do not use it as input of a node expecting tight boxes.
"""

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="SfMData file, used to match the frame numbers of the boxes with the images and blur maps.",
            value="",
        ),
        desc.File(
            name="bboxesFileIn",
            label="Bounding Boxes File",
            description="Input JSON file containing the bounding boxes.",
            value="",
        ),
        desc.File(
            name="blurMapFolder",
            label="Blur Map Folder",
            description="Output folder of an ImageBlurMap node. If empty, only the minimum margin is applied.",
            value="",
        ),
        desc.BoolParam(
            name="blurMapKeepFilename",
            label="Blur Maps Named By Filename",
            description="Must match the 'Keep Filename' option of the ImageBlurMap node "
                        "(blur maps named after the image filename, otherwise after the view id).",
            value=True,
        ),
        desc.IntParam(
            name="minBoxMargin",
            label="Min Box Margin",
            description="Margin in pixels added on every side of every box.",
            value=16,
            range=(0, 512, 1),
        ),
        desc.FloatParam(
            name="blurMarginFactor",
            label="Blur Margin Factor",
            description="Factor applied to the motion blur length to get the additional margin on every side.",
            value=1.0,
            range=(0.0, 5.0, 0.1),
        ),
        desc.FloatParam(
            name="blurPercentile",
            label="Blur Percentile",
            description="Percentile of the motion blur length inside the box used to compute the margin.",
            value=95.0,
            range=(50.0, 100.0, 1.0),
            advanced=True,
        ),
        desc.BoolParam(
            name="outputBboxImage",
            label="Output Bounding Box Image",
            description="Write the source images with the input boxes (red) and the expanded boxes (green).",
            value=False,
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
            name="bboxesFile",
            label="Bounding Boxes File",
            description="JSON file containing the expanded bounding boxes.",
            value="{nodeCacheFolder}/bboxes.json",
        ),
        desc.File(
            name="bboxes",
            label="BBoxes",
            description="Images with the input (red) and expanded (green) boxes baked in.",
            semantic="image",
            value="{nodeCacheFolder}/bboxes_<FILESTEM>.jpg",
            enabled=lambda node: node.outputBboxImage.value,
        ),
    ]

    def processChunk(self, chunk):
        from pathlib import Path
        import numpy as np
        from pyalicevision import sfmData, sfmDataIO
        from pyalicevision import image as avimg
        from segmentationRDS import image, bboxUtils
        from segmentationRDS.mask_label_manager import BBoxesManager

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            node = chunk.node

            if not os.path.isfile(node.bboxesFileIn.value):
                raise FileNotFoundError(f"Bounding boxes file not found: {node.bboxesFileIn.value}")
            with open(node.bboxesFileIn.value, "r") as f:
                data = json.load(f)

            views = {}
            if node.input.value:
                dataAV = sfmData.SfMData()
                if not sfmDataIO.load(dataAV, node.input.value, sfmDataIO.ALL):
                    raise RuntimeError(f"Cannot load SfMData: {node.input.value}")
                for vId, v in dataAV.getViews().items():
                    views[v.getFrameId()] = {"path": v.getImage().getImagePath(), "viewId": str(vId)}

            blurFolder = node.blurMapFolder.value
            cache = {"frame": None, "map": None}
            missing = []

            def blurMapOf(frameIdx):
                if cache["frame"] != frameIdx:
                    cache["frame"], cache["map"] = frameIdx, None
                    view = views.get(frameIdx)
                    if blurFolder and view is None:
                        missing.append(frameIdx)  # frame number not found in the SfMData
                    elif blurFolder:
                        name = Path(view["path"]).stem if node.blurMapKeepFilename.value else view["viewId"]
                        path = os.path.join(blurFolder, name + "_motionBlurLength.exr")
                        if os.path.exists(path):
                            cache["map"] = image.loadImage(path, applyPAR=False, srgb=False, clamp=False)[0][:, :, 0]
                        else:
                            missing.append(frameIdx)
                return cache["map"]

            def marginOf(frameIdx, box):
                m = node.minBoxMargin.value
                blur = blurMapOf(frameIdx)
                if blur is not None:
                    x1, y1, x2, y2 = (max(0, int(v)) for v in box)
                    crop = blur[y1:y2, x1:x2]
                    if crop.size:
                        m += node.blurMarginFactor.value * float(np.percentile(crop, node.blurPercentile.value))
                return m

            def sizeOf(frameIdx):
                blur = blurMapOf(frameIdx)
                if blur is not None:
                    return blur.shape[1], blur.shape[0]
                # Boxes are in the oriented space: the SfMData size is the stored (unoriented) one.
                return image.orientedSize(views[frameIdx]["path"]) if frameIdx in views else None

            out, margins = bboxUtils.expand_boxes(data, marginOf, sizeOf)

            if blurFolder and missing:
                chunk.logger.warning(f"{len(missing)} blur map(s) not found (missing file or frame number not in the SfMData), minimum margin only for frames {missing[:10]}...")
            if blurFolder and not views:
                chunk.logger.warning("No SfMData provided: blur maps cannot be matched, minimum margin only.")
            for key, values in margins.items():
                chunk.logger.info(f"{key}: margin min {min(values):.1f} / mean {np.mean(values):.1f} / max {max(values):.1f} px")

            with open(node.bboxesFile.value, "w") as f:
                json.dump(out, f, indent=4)

            if node.outputBboxImage.value:
                # loadImage returns sRGB values: write them as is, writeImage would otherwise apply linear->sRGB again.
                optWrite = avimg.ImageWriteOptions()
                optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
                for frameIdx, view in sorted(views.items()):
                    img, h, w, par, orientation = image.loadImage(view["path"], applyPAR=False)
                    drawn = False
                    for label, tracks in data.items():
                        tkey = BBoxesManager.tracking_key_of(tracks)
                        for frameKey, objs in tracks.get(tkey, {}).items():
                            if int(frameKey) != frameIdx:
                                continue
                            for objId, box in objs.items():
                                if box:
                                    img = image.addRectangle(img, box, (255, 0, 0))
                                    img = image.addRectangle(img, out[label][tkey][frameKey][objId], (0, 255, 0))
                                    drawn = True
                    if drawn:
                        outPath = os.path.join(os.path.dirname(node.bboxesFile.value), "bboxes_" + Path(view["path"]).stem + ".jpg")
                        image.writeImage(outPath, img, h, w, orientation, par, optWrite=optWrite)
        finally:
            chunk.logManager.end()
