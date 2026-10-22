"""FastAPI service for video-based traffic congestion prediction."""

import json
import pickle
import tempfile
from collections import deque
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from sklearn.preprocessing import MinMaxScaler

from core.preprocessing import preprocess_frame


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
WINDOW_FRAMES = 30
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
TRACK_DISTANCE_THRESHOLD = 50.0
MAX_TRACK_AGE = 5
SUPPORTED_VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
COCO_CLASS_IDS = {"car": 2, "truck": 7, "bus": 5}


@dataclass
class Detection:
    """A selected vehicle detection in resized-frame coordinates."""

    class_id: int
    box: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        """Return the center point of the bounding box."""
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0


@dataclass
class Track:
    """Last observed location and class for a centroid-matched vehicle."""

    track_id: int
    class_id: int
    center: tuple[float, float]
    last_frame: int


def _read_manifest() -> Mapping[str, Any]:
    """Read the exported model manifest and validate its required fields."""
    try:
        with MANIFEST_PATH.open(encoding="utf-8") as manifest_file:
            manifest: Any = json.load(manifest_file)
    except FileNotFoundError as error:
        raise RuntimeError(f"Model manifest not found: {MANIFEST_PATH}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Model manifest is not valid JSON: {MANIFEST_PATH}") from error

    if not isinstance(manifest, Mapping):
        raise ValueError("Model manifest must contain a JSON object")
    if not isinstance(manifest.get("version"), str) or not manifest["version"]:
        raise ValueError("Model manifest must define a non-empty version")
    sequence_length = manifest.get("sequence_length")
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length <= 0
    ):
        raise ValueError("Model manifest sequence_length must be a positive integer")
    classes = manifest.get("classes")
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(class_name, str) for class_name in classes)
    ):
        raise ValueError("Model manifest classes must be a non-empty list of strings")
    models = manifest.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("Model manifest must define a models object")
    for model_name in ("detector", "predictor", "scaler"):
        if not isinstance(models.get(model_name), str) or not models[model_name]:
            raise ValueError(f"Model manifest must define models.{model_name}")
    return manifest


def _model_path(relative_path: str) -> Path:
    """Resolve a manifest path relative to the project root."""
    path = Path(relative_path)
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {resolved}")
    return resolved


def _load_runtime() -> dict[str, Any]:
    """Load CPU ONNX sessions, scaler, and validated manifest metadata."""
    manifest = _read_manifest()
    models = manifest["models"]
    detector_path = _model_path(models["detector"])
    predictor_path = _model_path(models["predictor"])
    scaler_path = _model_path(models["scaler"])

    detector = ort.InferenceSession(
        str(detector_path),
        providers=["CPUExecutionProvider"],
    )
    predictor = ort.InferenceSession(
        str(predictor_path),
        providers=["CPUExecutionProvider"],
    )
    with scaler_path.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    if not isinstance(scaler, MinMaxScaler):
        raise TypeError(f"Expected a fitted MinMaxScaler in {scaler_path}")

    vehicle_classes = manifest.get("vehicle_classes", manifest["classes"])
    if (
        not isinstance(vehicle_classes, list)
        or not vehicle_classes
        or any(not isinstance(class_name, str) for class_name in vehicle_classes)
    ):
        raise ValueError("Model manifest vehicle_classes must be a non-empty list")

    unsupported_classes = [
        name for name in vehicle_classes if name not in COCO_CLASS_IDS
    ]
    if unsupported_classes:
        raise ValueError(
            "Detector class mapping is not configured for: "
            + ", ".join(unsupported_classes)
        )

    detector_input = detector.get_inputs()[0]
    predictor_input = predictor.get_inputs()[0]
    predictor_output = predictor.get_outputs()[0]

    return {
        "manifest": manifest,
        "detector": detector,
        "predictor": predictor,
        "scaler": scaler,
        "detector_input_name": detector_input.name,
        "predictor_input_name": predictor_input.name,
        "predictor_output_name": predictor_output.name,
        "class_ids": {COCO_CLASS_IDS[name] for name in vehicle_classes},
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize model artifacts once when the API service starts."""
    app.state.runtime = _load_runtime()
    yield


app = FastAPI(
    title="Traffic Congestion Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


def _detect(
    detector: ort.InferenceSession,
    input_name: str,
    processed_frame: np.ndarray,
    original_shape: tuple[int, int],
    class_ids: set[int],
) -> list[Detection]:
    """Run the YOLO ONNX model and decode common YOLOv8 output layouts."""
    input_tensor = np.ascontiguousarray(
        processed_frame[:, :, ::-1].transpose(2, 0, 1)[np.newaxis, ...],
        dtype=np.float32,
    )
    output = np.asarray(detector.run(None, {input_name: input_tensor})[0])
    output = np.squeeze(output)
    if output.ndim != 2:
        raise RuntimeError(f"Unexpected detector output shape: {output.shape}")

    if output.shape[0] < output.shape[1] and output.shape[0] != 6:
        output = output.T

    boxes: list[tuple[float, float, float, float]] = []
    scores: list[float] = []
    detection_classes: list[int] = []

    if output.shape[1] == 6:
        for row in output:
            class_id = int(row[5])
            score = float(row[4])
            if class_id in class_ids and score >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = (float(value) for value in row[:4])
                boxes.append((x1, y1, x2, y2))
                scores.append(score)
                detection_classes.append(class_id)
    elif output.shape[1] > 6:
        for row in output:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if class_id not in class_ids or score < CONFIDENCE_THRESHOLD:
                continue

            center_x, center_y, box_width, box_height = (
                float(value) for value in row[:4]
            )
            boxes.append(
                (
                    center_x - box_width / 2.0,
                    center_y - box_height / 2.0,
                    center_x + box_width / 2.0,
                    center_y + box_height / 2.0,
                )
            )
            scores.append(score)
            detection_classes.append(class_id)
    else:
        raise RuntimeError(f"Unexpected detector output shape: {output.shape}")

    kept_indices: list[int] = []
    for class_id in class_ids:
        class_indices = [
            index
            for index, detected_class in enumerate(detection_classes)
            if detected_class == class_id
        ]
        if not class_indices:
            continue
        xywh_boxes = [
            [
                boxes[index][0],
                boxes[index][1],
                boxes[index][2] - boxes[index][0],
                boxes[index][3] - boxes[index][1],
            ]
            for index in class_indices
        ]
        nms_indices = cv2.dnn.NMSBoxes(
            xywh_boxes,
            [scores[index] for index in class_indices],
            CONFIDENCE_THRESHOLD,
            NMS_THRESHOLD,
        )
        kept_indices.extend(
            class_indices[int(index)]
            for index in np.asarray(nms_indices).reshape(-1)
        )

    original_height, original_width = original_shape
    scale_x = original_width / processed_frame.shape[1]
    scale_y = original_height / processed_frame.shape[0]
    return [
        Detection(
            detection_classes[index],
            (
                boxes[index][0] * scale_x,
                boxes[index][1] * scale_y,
                boxes[index][2] * scale_x,
                boxes[index][3] * scale_y,
            ),
        )
        for index in kept_indices
    ]


def _update_tracks(
    detections: list[Detection],
    tracks: dict[int, Track],
    next_track_id: int,
    frame_index: int,
) -> tuple[dict[int, Track], int, list[tuple[int, Detection, float, int]]]:
    """Associate detections to tracks by nearest same-class centroid."""
    candidates: list[tuple[float, int, int]] = []
    for detection_index, detection in enumerate(detections):
        for track_id, track in tracks.items():
            if (
                track.class_id != detection.class_id
                or frame_index - track.last_frame > MAX_TRACK_AGE
            ):
                continue
            distance = float(np.hypot(
                detection.center[0] - track.center[0],
                detection.center[1] - track.center[1],
            ))
            if distance <= TRACK_DISTANCE_THRESHOLD:
                candidates.append((distance, detection_index, track_id))

    candidates.sort()
    assigned_detections: set[int] = set()
    assigned_tracks: set[int] = set()
    associations: list[tuple[int, Detection, float, int]] = []
    updated_tracks: dict[int, Track] = {}

    for distance, detection_index, track_id in candidates:
        if detection_index in assigned_detections or track_id in assigned_tracks:
            continue
        detection = detections[detection_index]
        previous = tracks[track_id]
        elapsed_frames = frame_index - previous.last_frame
        updated_tracks[track_id] = Track(
            track_id=track_id,
            class_id=detection.class_id,
            center=detection.center,
            last_frame=frame_index,
        )
        associations.append((track_id, detection, distance, elapsed_frames))
        assigned_detections.add(detection_index)
        assigned_tracks.add(track_id)

    for track_id, track in tracks.items():
        if track_id not in assigned_tracks and frame_index - track.last_frame <= MAX_TRACK_AGE:
            updated_tracks[track_id] = track

    for detection_index, detection in enumerate(detections):
        if detection_index in assigned_detections:
            continue
        track_id = next_track_id
        next_track_id += 1
        updated_tracks[track_id] = Track(
            track_id=track_id,
            class_id=detection.class_id,
            center=detection.center,
            last_frame=frame_index,
        )
        associations.append((track_id, detection, 0.0, 0))

    return updated_tracks, next_track_id, associations


def _predict_window(
    runtime: Mapping[str, Any],
    feature_sequence: deque[np.ndarray],
) -> int:
    """Scale a feature sequence and return the predictor's congestion class."""
    raw_features = np.stack(feature_sequence).astype(np.float32)
    scaled_features = runtime["scaler"].transform(raw_features).astype(np.float32)
    sequence = np.ascontiguousarray(scaled_features[np.newaxis, ...])
    logits = runtime["predictor"].run(
        [runtime["predictor_output_name"]],
        {runtime["predictor_input_name"]: sequence},
    )[0]
    return int(np.argmax(np.asarray(logits), axis=-1).reshape(-1)[0])


def _process_video(video_path: Path, runtime: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract windowed vehicle features and predict congestion levels."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("Uploaded file could not be opened as a video")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Uploaded video has invalid frame-rate metadata")

    sequence_length = runtime["manifest"]["sequence_length"]
    feature_sequence: deque[np.ndarray] = deque(maxlen=sequence_length)
    results: list[dict[str, Any]] = []
    tracks: dict[int, Track] = {}
    next_track_id = 0
    frame_index = 0
    window_start = 0
    frame_area = 0
    bbox_area_sum = 0.0
    present_track_ids: set[int] = set()
    track_motion: dict[int, list[float]] = {}
    class_ids = runtime["class_ids"]

    def finish_window(window_frame_count: int) -> None:
        """Aggregate one feature row and predict once a full sequence exists."""
        speeds = [
            motion[0] / (motion[1] / fps)
            for motion in track_motion.values()
            if motion[1] > 0
        ]
        average_speed = float(np.mean(speeds)) if speeds else 0.0
        speed_cv = (
            float(np.std(speeds) / average_speed)
            if speeds and average_speed > 0
            else 0.0
        )
        occupancy = (
            bbox_area_sum / (frame_area * window_frame_count)
            if frame_area > 0 and window_frame_count > 0
            else 0.0
        )
        features = np.asarray(
            [
                len(present_track_ids),
                average_speed,
                speed_cv,
            ],
            dtype=np.float32,
        )
        feature_sequence.append(features)
        if len(feature_sequence) == sequence_length:
            results.append(
                {
                    "timestamp": round(frame_index / fps, 3),
                    "congestion_level": _predict_window(runtime, feature_sequence),
                    "features": {
                        "car_count": len(present_track_ids),
                        "avg_speed": round(average_speed, 3),
                    },
                }
            )

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index > window_start and frame_index - window_start >= WINDOW_FRAMES:
                finish_window(frame_index - window_start)
                window_start = frame_index
                frame_area = 0
                bbox_area_sum = 0.0
                present_track_ids = set()
                track_motion = {}

            processed_frame = preprocess_frame(frame)
            frame_area = frame.shape[0] * frame.shape[1]
            detections = _detect(
                runtime["detector"],
                runtime["detector_input_name"],
                processed_frame,
                frame.shape[:2],
                class_ids,
            )
            tracks, next_track_id, associations = _update_tracks(
                detections,
                tracks,
                next_track_id,
                frame_index,
            )

            for track_id, detection, distance, elapsed_frames in associations:
                present_track_ids.add(track_id)
                x1, y1, x2, y2 = detection.box
                x1 = float(np.clip(x1, 0, frame.shape[1]))
                y1 = float(np.clip(y1, 0, frame.shape[0]))
                x2 = float(np.clip(x2, 0, frame.shape[1]))
                y2 = float(np.clip(y2, 0, frame.shape[0]))
                bbox_area_sum += max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if elapsed_frames > 0:
                    motion = track_motion.setdefault(track_id, [0.0, 0.0])
                    motion[0] += distance
                    motion[1] += elapsed_frames

            frame_index += 1
    finally:
        capture.release()

    if frame_index == 0:
        raise ValueError("Uploaded video contains no decodable frames")

    remaining_frames = frame_index - window_start
    if remaining_frames:
        finish_window(remaining_frames)
    return results


@app.post("/api/v1/predict")
def predict_video(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """Analyze an uploaded video and return windowed congestion predictions."""
    filename = file.filename or ""
    content_type = (file.content_type or "").lower()
    if (
        not content_type.startswith("video/")
        and Path(filename).suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS
    ):
        raise HTTPException(status_code=400, detail="Uploaded file must be a video")

    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Model runtime is not initialized")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        suffix = ".mp4"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(file.file.read())
            temporary_path = Path(temporary_file.name)

        predictions = _process_video(temporary_path, runtime)
    except (ValueError, cv2.error) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        file.file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "status": "success",
        "model_version": runtime["manifest"]["version"],
        "results": predictions,
    }
