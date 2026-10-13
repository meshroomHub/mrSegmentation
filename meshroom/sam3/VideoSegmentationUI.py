__version__ = "1.0"

import logging
import os
from pathlib import Path

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

logger = logging.getLogger("VideoSegmentationUI")

class VideoSegmentationUI(desc.Node):
    """
## Video Segmentation User Interface
"""
    category = "Segmentation"

    image_paths = []

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="SfMData file.",
            value="",
        ),
        desc.File(
            name="bboxesFileIn",
            label="Bounding Boxes File In",
            description="File containing mask bounding boxes (.json).",
            value="",
        ),
        desc.File(
            name="masksFolder",
            label="Masks Folder",
            description="Folder containing original masks.",
            value="",
        ),
        desc.PushButtonParam(
            name="LaunchUI",
            label="Launch UI",
            description="Launch the UI to update existing masks",
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
            name="bboxesFileOut",
            label="Bounding Boxes File Out",
            description="Output json file containing mask bounding boxes.",
            value="{nodeCacheFolder}/bboxes.json",
        ),
    ]

    def onLaunchUIClicked(self, node):
        import sys
        import shutil
        from PySide6.QtCore import QProcess

        update_gui = os.getenv("MRSEGMENTATION_UPDATE_GUI")
        sequence_folder = get_sequence_folder(node.input.value)
        if not sequence_folder:
            raise FileNotFoundError(f'No image files found in {node.input.value}')
        mask_folder = os.path.join(node.output.value, "colorMasks", "merged")
        if not os.path.isdir(mask_folder):
            mask_folder = os.path.join(node.output.value, "colorMasks", "fwd")
        json_path = os.path.join(node.output.value, "bboxes.json")

        args = [update_gui, sequence_folder, mask_folder, json_path]
        gui_folder = os.path.join(node.output.value, "gui")
        if not os.path.isdir(gui_folder):
            os.mkdir(gui_folder)
        else:
            clean_folder(gui_folder)

        p = QProcess()
        p.setProgram(sys.executable)
        p.setArguments(args)
        p.setWorkingDirectory(gui_folder)
        p.setStandardOutputFile(os.path.join(gui_folder, "gui_stdout.log"))
        p.setStandardErrorFile(os.path.join(gui_folder, "gui_stderr.log"))

        res = p.startDetached()
        if isinstance(res, tuple):
            ok, pid = res
        else:
            ok, pid = res, None
        if ok:
            msg = "UI launched"
            if pid:
                msg += f" with PID {pid}"
            print(msg)
        else:
            print(f"UI launch error : {p.errorString()}")

    def processChunk(self, chunk):
        import shutil
        src = os.path.join(chunk.node.masksFolder.value, "colorMasks")
        dest = os.path.join(chunk.node.output.value, "colorMasks")
        shutil.copytree(src, dest, dirs_exist_ok=True)
        src = os.path.join(chunk.node.masksFolder.value, "bboxes.json")
        dest = os.path.join(chunk.node.output.value, "bboxes.json")
        shutil.copyfile(src, dest)


def get_sequence_folder(input_path):
    from pyalicevision import sfmData, camera
    from pyalicevision import sfmDataIO
    from segmentationRDS.sfmData import isSfmDataFile

    sequence_folder = ""
    if isSfmDataFile(Path(input_path)):
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                commonParams = None
                for _, v in views.items():
                    image_x1_path = Path(v.getImage().getImagePath())
                    intrinsic = dataAV.getIntrinsicSharedPtr(v.getIntrinsicId())
                    pinhole = camera.Pinhole.cast(intrinsic)
                    par = 1.0
                    if pinhole is not None:
                        par = pinhole.getPixelAspectRatio()
                    sequence_folder = os.path.dirname(str(image_x1_path))
                    params = [sequence_folder, v.getImage().getWidth(), v.getImage().getHeight(), par]
                    if commonParams is None:
                        commonParams = params
                    if commonParams != params:
                        raise ValueError(f"All images do not have same dimensions or are not located in the same folder: {params} vs {commonParams}")
        else:
            raise ValueError(f"Input path '{input_path}' does not exist.")
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid sfmData file.")
    return sequence_folder

def clean_folder(path):
    import shutil
    if os.path.isdir(path):
        for elem in os.listdir(path):
            p = os.path.join(path, elem)
            try:
                if os.path.isfile(p) or os.path.islink(p):
                    os.unlink(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
            except Exception as e:
                print(f"Cannot remove {p}. Error {e}")
    else:
        raise ValueError(f"Input path '{path}' is not a folder.")
