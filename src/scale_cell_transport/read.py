"""
Read stuff from places
"""

import re
import pathlib
from collections import defaultdict

import tifffile
import numpy as np

from . import files


def _tif_paths(img_dir: pathlib.Path) -> list[str]:
    """
    Get the paths to the tif files in a given dir

    :param img_dir: the directory to search for tif files

    :returns: a list of paths to the tif files

    """
    return sorted(img_dir.glob("*.tif"))


def _group_file_names(file_paths: list[str]) -> dict[str, list[str]]:
    """
    Get a dict mapping the video name to the list of file paths
    """
    video_files = defaultdict(list)
    pattern = re.compile(r"(.+?)_\d+h\d+m\.tif$")

    for file_path in file_paths:
        match = pattern.match(file_path.name)
        if match:
            name = match.group(1)
            video_files[name].append(file_path)

    return video_files


def phase_videos() -> dict[str, np.ndarray]:
    """
    Get the greyscale phase contrast videos as a list of numpy arrays

    :returns: a dict mapping the video name to the video

    """
    img_dir = _tif_paths(files.incucyte_phase_imgs_dir())

    grouped_paths: dict[str, list[str]] = _group_file_names(img_dir)
