"""
Read stuff from places
"""

import re
import pathlib
import concurrent.futures
from functools import partial
from collections import defaultdict

import cv2
import tifffile
import numpy as np
from tqdm import tqdm

from . import files


def read_mp4s_from_dir(video_dir: pathlib.Path) -> list[np.ndarray]:
    """
    Read mp4 videos from a directory and return them as numpy arrays
    """

    arrays = []
    for path in tqdm(
        list(video_dir.glob("*.mp4")), desc=f"Loading videos from {video_dir.name}"
    ):
        cap = cv2.VideoCapture(str(path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buffer = np.empty((n_frames, height, width, 3), dtype=np.uint8)

        fc = 0
        ret = True

        while fc < n_frames and ret:
            ret, buffer[fc] = cap.read()
            fc += 1

        # Convert BGR to RGB
        arrays.append(buffer[:, :, :, ::-1])
        cap.release()


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


def phase_videos() -> dict[str, tuple[np.ndarray, list[str]]]:
    """
    Get the greyscale phase contrast videos as a list of numpy arrays

    :returns: a dict mapping the video name to the video and timestamp of each frame

    """
    img_dir = files.incucyte_phase_imgs_dir()

    grouped_paths = _group_file_names(_tif_paths(img_dir))

    videos = {}

    def read_tif(path: pathlib.Path) -> np.ndarray:
        """
        Read a tif file and return it as a numpy array
        """
        return tifffile.imread(path)

    for video_name, file_data in tqdm(
        grouped_paths.items(), desc=f"Loading videos from {img_dir.name}"
    ):
        paths, times = zip(*file_data)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Read the tif files in parallel
            frames = list(executor.map(read_tif, paths))

        if frames:
            videos[video_name] = (np.stack(frames, axis=0), times)
        else:
            raise ValueError(f"No frames found for video {video_name}")

    return videos
