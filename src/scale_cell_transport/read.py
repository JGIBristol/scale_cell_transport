"""
Read stuff from places
"""

import re
import pathlib
from collections import defaultdict

import tifffile
import numpy as np
from tqdm import tqdm

from . import files


def _tif_paths(img_dir: pathlib.Path) -> list[str]:
    """
    Get the paths to the tif files in a given dir

    :param img_dir: the directory to search for tif files

    :returns: a list of paths to the tif files

    """
    return sorted(img_dir.glob("*.tif"))


def _group_file_names(
    file_paths: list[pathlib.Path],
) -> dict[str, list[tuple[pathlib.Path, str]]]:
    """
    Get a dict mapping the video name to the list of file paths
    """
    # defaultdict so that we don't need to check for existence every time
    video_files = defaultdict(list)

    # Extract video name and the time
    pattern = re.compile(r"(.*?)_((?:\d+y\d+m\d+d)_\d+h\d+m)\.tif$")

    for file_path in file_paths:
        match = pattern.match(file_path.name)
        if match:
            name = match.group(1)
            timestamp = match.group(2)

            video_files[name].append((file_path, timestamp))

    return video_files


def phase_videos() -> dict[str, np.ndarray]:
    """
    Get the greyscale phase contrast videos as a list of numpy arrays

    :returns: a dict mapping the video name to the video

    """
    img_dir = files.incucyte_phase_imgs_dir()

    grouped_paths = _group_file_names(_tif_paths(img_dir))

    videos = {}
    for video_name, file_data in tqdm(
        grouped_paths.items(), desc=f"Loading videos from {img_dir.name}"
    ):
        path, time = zip(*file_data)

        frames = [tifffile.imread(p) for p in path]

        if frames:
            # Stack the frames into a 3D array
            videos[video_name] = np.stack(frames, axis=0)

    return videos
