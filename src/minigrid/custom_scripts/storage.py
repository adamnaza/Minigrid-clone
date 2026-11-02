"""Shared output location for custom training and evaluation scripts."""

import os
from pathlib import Path


def get_storage_dir():
    """Use RL_STORAGE, or runs/ at the source checkout root.

    Installed packages without a project marker fall back to runs/ in the
    current working directory.
    """
    if "RL_STORAGE" in os.environ:
        return Path(os.environ["RL_STORAGE"])
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent / "runs"
    return Path.cwd() / "runs"
