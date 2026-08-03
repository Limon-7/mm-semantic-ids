"""Data locations. Everything hangs off the RS_DATA environment variable."""

import os
from pathlib import Path


def data_root() -> Path:
    root = os.environ.get("RS_DATA")
    if not root:
        raise RuntimeError("RS_DATA is not set. Export it to the folder that holds the data.")
    return Path(root)


def raw_dir() -> Path:
    return data_root() / "raw"


def processed_dir(dataset: str, split: str | None = None) -> Path:
    p = data_root() / "processed" / dataset
    return p / split if split else p


def features_dir(dataset: str) -> Path:
    return data_root() / "features" / dataset


def checkpoints_dir() -> Path:
    return data_root() / "checkpoints"
