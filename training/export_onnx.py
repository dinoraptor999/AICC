"""Export the traffic predictor and YOLO detector to ONNX."""

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

if __package__:
    from training.train_predictor import TrafficLSTM, load_sequence_length
else:
    from train_predictor import TrafficLSTM, load_sequence_length


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
PREDICTOR_WEIGHTS_PATH = MODELS_DIR / "predictor.pth"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
PREDICTOR_ONNX_PATH = MODELS_DIR / "predictor.onnx"
DETECTOR_ONNX_PATH = MODELS_DIR / "detector.onnx"
MANIFEST_PATH = MODELS_DIR / "manifest.json"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
PREDICTOR_CLASSES = ["Thong_thoang", "Dong_duc"]
PREDICTOR_FEATURES = ["car_count", "avg_speed", "speed_cv"]


def load_vehicle_classes() -> list[str]:
    """Read detector vehicle classes from the project configuration."""
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config: Any = yaml.safe_load(config_file)

    if not isinstance(config, Mapping):
        raise ValueError(f"Expected a YAML mapping in {CONFIG_PATH}")

    classes = config.get("vehicle_classes")
    if (
        not isinstance(classes, list)
        or not classes
        or any(not isinstance(class_name, str) or not class_name for class_name in classes)
    ):
        raise ValueError("config.yaml must define vehicle_classes as a non-empty list")
    return classes


def export_predictor(sequence_length: int) -> None:
    """Export trained LSTM weights to ONNX with a dynamic batch dimension."""
    if not PREDICTOR_WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Predictor weights not found: {PREDICTOR_WEIGHTS_PATH}. "
            "Train the predictor before exporting."
        )
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(
            f"Fitted scaler not found: {SCALER_PATH}. "
            "Run training/prepare_dataset.py before exporting."
        )

    model = TrafficLSTM(input_size=3)
    state_dict = torch.load(
        PREDICTOR_WEIGHTS_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, sequence_length, 3)
    torch.onnx.export(
        model,
        dummy_input,
        str(PREDICTOR_ONNX_PATH),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        dynamo=False,
    )


def export_detector() -> None:
    """Export the same pretrained YOLO model used during feature extraction."""
    detector = YOLO("yolov8n.pt")
    exported_path = Path(detector.export(format="onnx"))
    if not exported_path.is_file():
        raise FileNotFoundError(
            f"Ultralytics reported an ONNX export that does not exist: {exported_path}"
        )

    if exported_path.resolve() != DETECTOR_ONNX_PATH.resolve():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported_path, DETECTOR_ONNX_PATH)
        exported_path.unlink()


def write_manifest(sequence_length: int, vehicle_classes: list[str]) -> None:
    """Write predictor metadata and paths to deployed artifacts."""
    manifest = {
        "version": "v1.0-car-only",
        "classes": PREDICTOR_CLASSES,
        "sequence_length": sequence_length,
        "features": PREDICTOR_FEATURES,
        "vehicle_classes": vehicle_classes,
        "models": {
            "predictor": "models/predictor.onnx",
            "detector": "models/detector.onnx",
            "scaler": SCALER_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Export predictor and detector models, then write their manifest."""
    sequence_length = load_sequence_length()
    vehicle_classes = load_vehicle_classes()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    export_predictor(sequence_length)
    export_detector()
    write_manifest(sequence_length, vehicle_classes)

    print(f"Exported predictor to {PREDICTOR_ONNX_PATH}")
    print(f"Exported detector to {DETECTOR_ONNX_PATH}")
    print(f"Scaler available at {SCALER_PATH}")
    print(f"Wrote manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
