"""Prepare chronological train and test sequences from master feature data."""

import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
INPUT_PATH = PROJECT_ROOT / "data" / "master_features.csv"
TRAIN_PATH = PROJECT_ROOT / "data" / "train.npz"
TEST_PATH = PROJECT_ROOT / "data" / "test.npz"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"

SEQUENCE_LENGTH = 5
FEATURE_COLUMNS = ("car_count", "avg_speed", "speed_cv")
REQUIRED_COLUMNS = ("video_id", "timestamp", *FEATURE_COLUMNS)
MAX_AVG_SPEED = 120.0
CONGESTED_CAR_COUNT = 9
CONGESTED_SPEED_MAX = 150.0
TRAIN_RATIO = 0.8


def create_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows with labels for the immediately following step.

    Args:
        features: Feature rows for one video partition, shape ``(rows, 3)``.
        labels: Integer labels aligned with ``features``, shape ``(rows,)``.
        sequence_length: Number of time steps in each sequence.

    Returns:
        ``X`` with shape ``(N, sequence_length, 3)`` and ``y`` with shape
        ``(N, 1)``. If the partition is too short, returns correctly shaped
        empty arrays.
    """
    if len(features) != len(labels):
        raise ValueError("features and labels must contain the same number of rows")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")

    sample_count = max(0, len(features) - sequence_length)
    if sample_count == 0:
        return (
            np.empty((0, sequence_length, len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0, 1), dtype=np.int64),
        )

    X = np.stack(
        [features[index : index + sequence_length] for index in range(sample_count)]
    ).astype(np.float32, copy=False)
    y = labels[sequence_length:].reshape(-1, 1).astype(np.int64, copy=False)
    return X, y


def load_sequence_length() -> int:
    """Read and validate sequence_length from the root YAML config."""
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config: Any = yaml.safe_load(config_file)
    if not isinstance(config, Mapping):
        raise ValueError(f"Expected a YAML mapping in {CONFIG_PATH}")

    sequence_length = config.get("sequence_length")
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length <= 0
    ):
        raise ValueError("config.yaml must define sequence_length as a positive integer")
    return sequence_length


def prepare_dataset() -> tuple[Path, Path, Path]:
    """Clean, label, scale, split, and save sequences from all videos.

    Each video's rows are sorted by timestamp and split chronologically before
    sequences are generated, preventing sequences from crossing video or
    train/test boundaries. The scaler is fitted only on rows in the training
    partitions.

    Returns:
        Paths to the train archive, test archive, and fitted scaler.

    Raises:
        FileNotFoundError: If the master feature CSV does not exist.
        ValueError: If required data is missing, invalid, or too short to
            produce both train and test sequences.
    """
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Master feature CSV not found: {INPUT_PATH}")

    sequence_length = load_sequence_length()
    data = pd.read_csv(INPUT_PATH)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(
            f"{INPUT_PATH} is missing required columns: {', '.join(missing_columns)}"
        )
    if data.empty:
        raise ValueError(f"No feature rows found in {INPUT_PATH}")

    data = data.loc[:, list(REQUIRED_COLUMNS)].copy()
    data["video_id"] = data["video_id"].astype("string").str.strip()
    if data["video_id"].isna().any() or data["video_id"].eq("").any():
        raise ValueError("video_id must be present for every row")

    for column in ("timestamp", *FEATURE_COLUMNS):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    numeric_values = data.loc[:, ["timestamp", *FEATURE_COLUMNS]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric_values).all():
        raise ValueError("timestamp and feature columns must contain finite numbers")

    # Clip tracking outliers to the specified 120 km/h maximum.
    data["avg_speed"] = data["avg_speed"].clip(upper=MAX_AVG_SPEED)

    data["label"] = np.where(
        (data["car_count"] >= CONGESTED_CAR_COUNT)
        & (data["avg_speed"] <= CONGESTED_SPEED_MAX),
        1,
        0,
    ).astype(np.int64)
    label_counts = data["label"].value_counts().reindex([0, 1], fill_value=0)
    total_rows = len(data)
    print("Label distribution:")
    print(f"  0 (Thông thoáng): {label_counts.loc[0] / total_rows * 100:.2f}%")
    print(f"  1 (Đông đúc): {label_counts.loc[1] / total_rows * 100:.2f}%")

    ordered_groups: list[tuple[np.ndarray, np.ndarray, int]] = []
    training_feature_rows: list[np.ndarray] = []
    for _, video_rows in data.groupby("video_id", sort=False):
        video_rows = video_rows.sort_values("timestamp", kind="stable")
        feature_values = video_rows.loc[:, list(FEATURE_COLUMNS)].to_numpy(
            dtype=np.float32
        )
        label_values = video_rows["label"].to_numpy(dtype=np.int64)
        split_index = int(len(video_rows) * TRAIN_RATIO)
        if split_index <= sequence_length or split_index >= len(video_rows):
            raise ValueError(
                "Each video must have enough rows to produce chronological "
                f"train and test sequences with sequence_length={sequence_length}; "
                f"found {len(video_rows)} rows."
            )

        ordered_groups.append((feature_values, label_values, split_index))
        training_feature_rows.append(feature_values[:split_index])

    scaler = MinMaxScaler()
    scaler.fit(np.concatenate(training_feature_rows, axis=0))

    train_X_parts: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    test_X_parts: list[np.ndarray] = []
    test_y_parts: list[np.ndarray] = []

    for feature_values, label_values, split_index in ordered_groups:
        scaled_values = scaler.transform(feature_values).astype(np.float32)
        train_X, train_y = create_sequences(
            scaled_values[:split_index],
            label_values[:split_index],
            sequence_length,
        )
        test_context_start = max(0, split_index - sequence_length)
        test_X, test_y = create_sequences(
            scaled_values[test_context_start:],
            label_values[test_context_start:],
            sequence_length,
        )
        test_target_offset = split_index - test_context_start - sequence_length
        test_X = test_X[test_target_offset:]
        test_y = test_y[test_target_offset:]
        if len(train_X):
            train_X_parts.append(train_X)
            train_y_parts.append(train_y)
        if len(test_X):
            test_X_parts.append(test_X)
            test_y_parts.append(test_y)

    if not train_X_parts:
        raise ValueError(
            f"No training sequences can be formed with sequence_length={sequence_length}"
        )
    if not test_X_parts:
        raise ValueError(
            f"No test sequences can be formed with sequence_length={sequence_length}"
        )

    train_X = np.concatenate(train_X_parts, axis=0)
    train_y = np.concatenate(train_y_parts, axis=0)
    test_X = np.concatenate(test_X_parts, axis=0)
    test_y = np.concatenate(test_y_parts, axis=0)

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TRAIN_PATH, X=train_X, y=train_y)
    np.savez_compressed(TEST_PATH, X=test_X, y=test_y)
    with SCALER_PATH.open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)

    print(f"Train arrays: X={train_X.shape}, y={train_y.shape}")
    print(f"Test arrays:  X={test_X.shape}, y={test_y.shape}")
    print(f"Saved scaler to {SCALER_PATH}")
    return TRAIN_PATH, TEST_PATH, SCALER_PATH


if __name__ == "__main__":
    prepare_dataset()
