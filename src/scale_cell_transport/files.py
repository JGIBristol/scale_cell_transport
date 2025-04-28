"""
Where files live etc.
"""

import pathlib


def config_file() -> pathlib.Path:
    """
    Get the path to the config file.
    """
    return pathlib.Path(__file__).parents[2] / "config.yml"
