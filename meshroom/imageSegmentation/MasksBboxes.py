__version__ = "1.0"

import logging
import os

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

logger = logging.getLogger("MasksBboxes")

class MasksBboxes(desc.Node):

    category = 'Utils'
    documentation = '''Extract bounding boxes from a set of input masks'''

    inputs = [
        desc.File(
            name='maskFolder',
            label='Mask Folder',
            description='maskFolder',
            value='Folder containing the masks',
        ),
        desc.BoolParam(
            name='alphaOnly',
            label='Alpha Channel Only',
            description='''Only the alpha channel of RGBA images or single channel images will be considered. Error in case of RGB images.''',
            value=False
        ),
        desc.BoolParam(
            name='firstOnly',
            label='First Channel Only',
            description='''Only the R channel of RGB or RGBA images will be considered. Single channel images will be processed as usual.''',
            value=False
        ),
        desc.ChoiceParam(
            name='extension',
            label='Input File Extension',
            description='Input image file extension.',
            value='exr',
            values=['exr', 'png', 'jpg'],
            exclusive=True,
        ),
    ]

    outputs = [
        desc.File(
            name='outputFolder',
            label='outputFolder',
            description='outputFolder',
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name='bboxesFile',
            label='Bounding Boxes File',
            description='Generated json file containing the bounding boxes.',
            value="{nodeCacheFolder}/bboxes.json",
        ),
    ]

    def extract_frame_number(self, filepath, extension="exr"):
        import re

        allowed_extensions = {"exr", "jpg", "jpeg", "png"}
        ext_normalized = extension.lower()

        if ext_normalized not in allowed_extensions:
            raise ValueError(
                f"Unsupported extension: '{extension}'. "
                f"Valid extensions: {sorted(allowed_extensions)}"
            )

        FRAME_PATTERN = re.compile(rf'\.(\d+)\.{re.escape(ext_normalized)}$', re.IGNORECASE)
        match = FRAME_PATTERN.search(str(filepath))
        if not match:
            raise ValueError(f"Frame number cannot be extracted from {filepath}")
        return int(match.group(1))

    def list_exr_files_sorted(self, folder, extension="exr"):
        from pathlib import Path
        import sys

        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"{folder} folder does not exist !")

        candidates = list(folder.glob(f"*.{extension}"))

        files_with_frame = []
        for f in candidates:
            try:
                frame_num = self.extract_frame_number(f, extension)
                files_with_frame.append((frame_num, f))
            except ValueError:
                print(f"[WARN] Ignored file (no frame number): {f}", file=sys.stderr)
                continue

        if not files_with_frame:
            raise FileNotFoundError(f"No valid file found in {folder} with extension {extension})")

        files_with_frame.sort(key=lambda t: t[0])
        return [f for _, f in files_with_frame]


    def read_exr_channels(self, filepath):
        import OpenImageIO as oiio
        import numpy as np

        inp = oiio.ImageInput.open(str(filepath))
        if inp is None:
            raise IOError(f"{filepath} cannot be opened: ({oiio.geterror()})")

        spec = inp.spec()
        width = spec.width
        height = spec.height
        nchannels = spec.nchannels

        pixels = inp.read_image(format=oiio.FLOAT)
        inp.close()

        if pixels is None:
            raise IOError(f"{filepath} cannot be read")

        arr = np.array(pixels, dtype=np.float32).reshape(height, width, nchannels)
        return arr, nchannels


    def get_bbox_from_mask(self, mask, threshold=0.5):
        import numpy as np

        binary = mask > threshold
        if not np.any(binary):
            return None

        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return [int(x_min), int(y_min), int(x_max+1), int(y_max+1)]


    def process_exr_file(self, filepath, threshold=0.5, alpha_only=False, first_only=False):
        arr, nchannels = self.read_exr_channels(filepath)

        bboxes = {}

        if alpha_only:
            if nchannels == 1:
                mask = arr[:, :, 0]
                bbox = self.get_bbox_from_mask(mask, threshold=threshold)
                if bbox is not None:
                    bboxes["0"] = bbox

            elif nchannels == 4:
                mask = arr[:, :, 3]
                bbox = self.get_bbox_from_mask(mask, threshold=threshold)
                if bbox is not None:
                    bboxes["0"] = bbox

            else:
                raise ValueError(
                    f"alpha_only=True but image {filepath} has no alpha channel (RGB image)."
                )
            
        elif first_only:
            mask = arr[:, :, 0]
            bbox = self.get_bbox_from_mask(mask, threshold=threshold)
            if bbox is not None:
                bboxes["0"] = bbox

        else:
            max_channels = min(nchannels, 4)  # R, G, B, A -> "0", "1", "2", "3"
            for c in range(max_channels):
                mask = arr[:, :, c]
                bbox = self.get_bbox_from_mask(mask, threshold=threshold)
                if bbox is not None:
                    bboxes[str(c)] = bbox

        return bboxes


    def compute_frame_bboxes(self, filepaths, threshold=0.5, alpha_only=False, first_only=False):

        frame_bboxes = {}

        for filepath in filepaths:
            frame_number = self.extract_frame_number(filepath, self.node.extension.value)

            bboxes = self.process_exr_file(filepath, threshold=threshold, alpha_only=alpha_only, first_only=first_only)

            if bboxes:
                frame_bboxes[str(frame_number)] = bboxes

        return frame_bboxes


    def processChunk(self, chunk):
        import json

        sorted_paths = self.list_exr_files_sorted(chunk.node.maskFolder.value, extension=chunk.node.extension.value)

        frame_bboxes = {}

        for filepath in sorted_paths:
            frame_number = self.extract_frame_number(filepath, chunk.node.extension.value)
            bboxes = self.process_exr_file(filepath, threshold=0.5, alpha_only=chunk.node.alphaOnly.value, first_only=chunk.node.firstOnly.value)
            if bboxes:
                frame_bboxes[str(frame_number)] = bboxes

        result = {
            "object": {
                "forward": frame_bboxes,
                "backward": {},
                "merged": {}
            }
        }
        with open(chunk.node.bboxesFile.value, "w") as f:
            json.dump(result, f, indent=4, sort_keys=False)



