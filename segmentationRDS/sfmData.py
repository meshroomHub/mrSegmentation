"""SfMData file detection shared by the segmentation nodes."""
from pathlib import Path

# Extensions of the SfMData files that AliceVision can load (sfmDataIO.load)
SFMDATA_EXTENSIONS = (".sfm", ".json", ".abc", ".usd", ".usda", ".usdc")


def isSfmDataFile(path):
    """True if path is an SfMData file readable by AliceVision (e.g. the .usda written by SfMFilter)."""
    return Path(str(path)).suffix.lower() in SFMDATA_EXTENSIONS
