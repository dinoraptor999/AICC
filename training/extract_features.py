"""Batch-extract ROI-filtered vehicle traffic features from Dataset A videos."""

import argparse
import csv
import re
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(
    r"C:\Users\hungm\OneDrive\Desktop\AICC\data\AIC21_Track1_Vehicle_Counting"
)
VIDEO_DIR = DATASET_ROOT / "Dataset_A"
ROI_DIR = DATASET_ROOT / "ROIs"
FPS_STATS_PATH = VIDEO_DIR / "datasetA_vid_stats.txt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "master_features.csv"

# Demo limits: process only a few videos and a short segment of each video.
MAX_VIDEOS = 3
MAX_FRAMES = 10000
DEFAULT_WINDOW_FRAMES = 30
VEHICLE_CLASS_IDS = [2, 7, 5]  # COCO IDs: car, truck, bus.
FEATURE_COLUMNS = (
    "video_id",
    "timestamp",
    "car_count",
    "avg_speed",
    "occupancy",
    "speed_cv",
    "label",
)


def natural_sort_key(path: Path) -> list[int | str]:
    """Create a case-insensitive natural-sort key for camera video names."""
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def load_fps_by_video(stats_path: Path = FPS_STATS_PATH) -> dict[str, float]:
    """Read exact per-video frame rates from the dataset tab-separated file."""
    if not stats_path.is_file():
        raise FileNotFoundError(f"Video FPS metadata file not found: {stats_path}")

    fps_by_video: dict[str, float] = {}
    with stats_path.open(encoding="utf-8-sig", newline="") as stats_file:
        reader = csv.DictReader(stats_file, delimiter="\t")
        required_columns = {"vid_name", "fps"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"{stats_path} must contain tab-separated vid_name and fps columns"
            )

        for row in reader:
            video_name = (row.get("vid_name") or "").strip()
            fps_text = (row.get("fps") or "").strip()
            if not video_name:
                continue
            try:
                fps = float(Fraction(fps_text))
            except (ValueError, ZeroDivisionError) as error:
                raise ValueError(
                    f"Invalid FPS value {fps_text!r} for {video_name} in {stats_path}"
                ) from error
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"FPS for {video_name} must be positive")
            fps_by_video[video_name] = fps

    return fps_by_video


def camera_id_for_video(video_path: Path) -> str:
    """Extract camera identity, including names with dawn/rain suffixes."""
    match = re.match(r"(cam_\d+)(?:_|$)", video_path.stem, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(
            f"Cannot determine camera ID from video filename: {video_path.name}"
        )
    return match.group(1)


def load_roi_polygon(roi_path: Path) -> np.ndarray:
    """Read an ROI polygon whose coordinate pairs are written one per line."""
    if not roi_path.is_file():
        raise FileNotFoundError(f"ROI polygon not found: {roi_path}")

    points: list[tuple[int, int]] = []
    with roi_path.open(encoding="utf-8-sig") as roi_file:
        for line_number, line in enumerate(roi_file, start=1):
            line = line.strip()
            if not line:
                continue
            coordinates = [part.strip() for part in line.split(",")]
            if len(coordinates) != 2:
                raise ValueError(
                    f"Invalid ROI coordinate at {roi_path}:{line_number}: {line!r}"
                )
            try:
                points.append((int(coordinates[0]), int(coordinates[1])))
            except ValueError as error:
                raise ValueError(
                    f"Invalid ROI coordinate at {roi_path}:{line_number}: {line!r}"
                ) from error

    if len(points) < 3:
        raise ValueError(f"ROI polygon must contain at least three points: {roi_path}")
    return np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))


def center_is_in_roi(center: tuple[float, float], polygon: np.ndarray) -> bool:
    """Return whether a vehicle center lies inside or on the ROI boundary."""
    return cv2.pointPolygonTest(polygon, center, False) >= 0


def extract_video_features(
    model: YOLO,
    video_path: Path,
    roi_polygon: np.ndarray,
    fps: float,
    window_frames: int,
) -> list[dict[str, float | int | str]]:
    """Extract ROI-filtered vehicle features from one video segment.

    ``avg_speed`` is measured in source-frame pixels per second, based on
    tracked vehicle center movement while the center is within the ROI.
    Occupancy is the total qualifying bounding-box area divided by the total
    frame area over the window. The final partial window is retained.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    rows: list[dict[str, float | int | str]] = []
    frame_index = 0
    window_start = 0
    tracked_ids: set[int] = set()
    track_motion: dict[int, dict[str, float]] = {}
    bbox_area_sum = 0.0
    frame_area_sum = 0
    video_id = video_path.stem

    def finish_window() -> None:
        """Aggregate the current window into a master CSV row."""
        speeds = [
            motion["distance"] / (motion["elapsed_frames"] / fps)
            for motion in track_motion.values()
            if motion["elapsed_frames"] > 0
        ]
        average_speed = float(np.mean(speeds)) if speeds else 0.0
        speed_cv = (
            float(np.std(speeds) / average_speed)
            if speeds and average_speed > 0
            else 0.0
        )
        occupancy = bbox_area_sum / frame_area_sum if frame_area_sum > 0 else 0.0
        rows.append(
            {
                "video_id": video_id,
                "timestamp": window_start / fps,
                "car_count": len(tracked_ids),
                "avg_speed": average_speed,
                "occupancy": occupancy,
                "speed_cv": speed_cv,
                "label": "",
            }
        )

    try:
        progress = tqdm(
            total=MAX_FRAMES,
            desc=video_id,
            unit="frame",
            leave=False,
        )
        try:
            while frame_index < MAX_FRAMES:
                success, frame = capture.read()
                if not success:
                    break

                if frame_index > window_start and frame_index - window_start >= window_frames:
                    finish_window()
                    window_start = frame_index
                    tracked_ids = set()
                    track_motion = {}
                    bbox_area_sum = 0.0
                    frame_area_sum = 0

                height, width = frame.shape[:2]
                frame_area_sum += height * width
                result = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=VEHICLE_CLASS_IDS,
                    verbose=False,
                )[0]

                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    ids = boxes.id.int().cpu().tolist()
                    coordinates = boxes.xyxy.cpu().numpy()

                    for track_id_value, (x1, y1, x2, y2) in zip(ids, coordinates):
                        left = float(np.clip(x1, 0, width))
                        top = float(np.clip(y1, 0, height))
                        right = float(np.clip(x2, 0, width))
                        bottom = float(np.clip(y2, 0, height))
                        center = ((left + right) / 2.0, (top + bottom) / 2.0)
                        if not center_is_in_roi(center, roi_polygon):
                            continue

                        track_id = int(track_id_value)
                        tracked_ids.add(track_id)
                        bbox_area_sum += max(0.0, right - left) * max(
                            0.0, bottom - top
                        )
                        motion = track_motion.get(track_id)
                        if motion is None:
                            track_motion[track_id] = {
                                "x": center[0],
                                "y": center[1],
                                "frame": float(frame_index),
                                "distance": 0.0,
                                "elapsed_frames": 0.0,
                            }
                        else:
                            elapsed_frames = frame_index - motion["frame"]
                            if elapsed_frames > 0:
                                motion["distance"] += float(
                                    np.hypot(
                                        center[0] - motion["x"],
                                        center[1] - motion["y"],
                                    )
                                )
                                motion["elapsed_frames"] += elapsed_frames
                            motion["x"] = center[0]
                            motion["y"] = center[1]
                            motion["frame"] = float(frame_index)

                frame_index += 1
                progress.update(1)
        finally:
            progress.close()
    finally:
        capture.release()

    if frame_index == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")

    finish_window()
    return rows


def extract_batch(window_frames: int = DEFAULT_WINDOW_FRAMES) -> Path:
    """Process the first ``MAX_VIDEOS`` dataset videos and save one CSV."""
    if window_frames <= 0:
        raise ValueError("window_frames must be a positive integer")
    if not VIDEO_DIR.is_dir():
        raise FileNotFoundError(f"Video dataset directory not found: {VIDEO_DIR}")

    videos = sorted(VIDEO_DIR.glob("*.mp4"), key=natural_sort_key)[:MAX_VIDEOS]
    if not videos:
        raise FileNotFoundError(f"No MP4 videos found in {VIDEO_DIR}")
    fps_by_video = load_fps_by_video()
    all_rows: list[dict[str, float | int | str]] = []

    for video_path in tqdm(videos, desc="Videos", unit="video"):
        camera_id = camera_id_for_video(video_path)
        roi_path = ROI_DIR / f"{camera_id}.txt"
        roi_polygon = load_roi_polygon(roi_path)
        fps = fps_by_video.get(video_path.name)
        if fps is None:
            raise ValueError(
                f"No FPS entry for {video_path.name} in {FPS_STATS_PATH}"
            )

        # A fresh model per video ensures ByteTrack IDs cannot leak between videos.
        model = YOLO("yolov8n.pt")
        all_rows.extend(
            extract_video_features(
                model=model,
                video_path=video_path,
                roi_polygon=roi_polygon,
                fps=fps,
                window_frames=window_frames,
            )
        )
        del model

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows, columns=FEATURE_COLUMNS).to_csv(OUTPUT_PATH, index=False)
    return OUTPUT_PATH


def main() -> None:
    """Parse batch options, process demo videos, and report the CSV location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-frames",
        type=int,
        default=DEFAULT_WINDOW_FRAMES,
        help=f"Frames per feature window (default: {DEFAULT_WINDOW_FRAMES})",
    )
    args = parser.parse_args()
    output_path = extract_batch(window_frames=args.window_frames)
    print(f"Saved batch features to {output_path}")


if __name__ == "__main__":
    main()
