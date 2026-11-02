"""Shared output location for custom training and evaluation scripts."""

import os
from pathlib import Path

def __get_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent

    working_dir = Path.cwd()
    print(f"WARNING! Failed to find project root directory." +
          f"\nMaybe the project root `pyproject.toml` file was moved?" +
          f"\nUsing working directory instead: '{working_dir}'")
    return working_dir

PROJECT_ROOT: Path = __get_project_root()


def get_storage_dir() -> Path:
    """Use RL_STORAGE, or runs/ at the source checkout root.

    Installed packages without a project marker fall back to runs/ in the
    current working directory.
    """
    if "RL_STORAGE" in os.environ:
        return Path(os.environ["RL_STORAGE"])
    
    return PROJECT_ROOT / "runs"