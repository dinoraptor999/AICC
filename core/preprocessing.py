"""Shared image preprocessing for model training and serving."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _load_preprocessing_settings() -> tuple[tuple[int, int], bool]:
    """Load and validate resize and normalization settings from config.yaml."""
    with _CONFIG_PATH.open(encoding="utf-8") as config_file:
        config: Any = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {_CONFIG_PATH}")

    preprocessing = config.get("preprocessing")
    if not isinstance(preprocessing, dict):
        raise ValueError("config.yaml must define preprocessing as a mapping")

    resize = preprocessing.get("resize")
    if (
        not isinstance(resize, list)
        or len(resize) != 2
        or any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in resize)
    ):
        raise ValueError("preprocessing.resize must contain two positive integers")

    normalize = preprocessing.get("normalize")
    if not isinstance(normalize, bool):
        raise ValueError("preprocessing.normalize must be a boolean")

    height, width = resize
    return (height, width), normalize


_RESIZE, _NORMALIZE = _load_preprocessing_settings()


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize and optionally normalize a video frame using ``config.yaml``.

    The configured ``resize`` is interpreted as ``[height, width]``. OpenCV's
    BGR channel order is preserved. When normalization is enabled, pixel values
    are converted to ``float32`` in the range [0, 1].

    Args:
        frame: Non-empty image as a NumPy array, typically read by OpenCV.

    Returns:
        The resized frame, optionally normalized to [0, 1].

    Raises:
        ValueError: If ``frame`` is empty or is not a two- or three-dimensional
            image array.
    """
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame must be a non-empty NumPy array")
    if frame.ndim not in (2, 3):
        raise ValueError("frame must be a two- or three-dimensional image array")

    height, width = _RESIZE
    processed = cv2.resize(frame, (width, height))

    if _NORMALIZE:
        processed = processed.astype(np.float32) / 255.0

    return processed
