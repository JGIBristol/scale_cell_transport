"""
General utilities
"""

from typing import Any

import yaml

from . import files


def config() -> dict[str, Any]:
    """
    Get the config file as a dictionary.
    """
    with open(files.config_file(), "r") as f:
        return yaml.safe_load(f)
