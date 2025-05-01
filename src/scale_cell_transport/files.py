"""
Where files live etc.
"""

import pathlib

from . import util


def config_file() -> pathlib.Path:
    """
    Get the path to the config file.
    """
    return pathlib.Path(__file__).parents[2] / "config.yml"


def rdsf_path() -> pathlib.Path:
    """
    Path to the RDSF directory (i.e. the mount location in the config file)
    """
    return pathlib.Path(util.config()["rdsf_dir"])


def model_path() -> pathlib.Path:
    """
    Path to the model directory
    """
    return (
        pathlib.Path(__file__).parents[2]
        / "src"
        / "rotir"
        / "Fishscale_registration_11454726_40000.pth"
    )


def incucyte_video_dir_1() -> pathlib.Path:
    """
    Path to directory containing some scale videos

    This lives on the RDSF, so it needs to be mounted and the mount location
    specified in the config file.

    """
    return rdsf_path() / "Jérémie" / "Scale Culture" / "video Incucyte for Rich"


def incucyte_phase_imgs_dir() -> pathlib.Path:
    """
    Path to directory containing some scale videos

    This lives on the RDSF, so it needs to be mounted and the mount location
    specified in the config file.

    """
    return (
        rdsf_path() / "Jérémie" / "Scale Culture" / "video Incucyte_37C_Phase_ for Rich"
    )
