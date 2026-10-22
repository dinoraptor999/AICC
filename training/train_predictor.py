"""Train a class-weighted LSTM traffic-congestion predictor."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
TRAIN_PATH = PROJECT_ROOT / "data" / "train.npz"
TEST_PATH = PROJECT_ROOT / "data" / "test.npz"
MODEL_PATH = PROJECT_ROOT / "models" / "predictor.pth"
FEATURE_COUNT = 3
CLASS_COUNT = 2
CLASS_LABELS = np.array([0, 1], dtype=np.int64)
HIDDEN_SIZE = 64
LAYER_COUNT = 2
EPOCH_COUNT = 50
BATCH_SIZE = 32


class TrafficLSTM(nn.Module):
    """Two-layer LSTM that classifies traffic as clear or congested."""

    def __init__(self, input_size: int = FEATURE_COUNT) -> None:
        """Initialize the LSTM and its binary output layer."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYER_COUNT,
            batch_first=True,
        )
        self.classifier = nn.Linear(HIDDEN_SIZE, CLASS_COUNT)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """Return class logits using the final step of each input sequence."""
        outputs, _ = self.lstm(sequence)
        return self.classifier(outputs[:, -1, :])


def load_sequence_length() -> int:
    """Read and validate ``sequence_length`` from the project configuration."""
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


def load_dataset(path: Path, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate an ``X``/``y`` sequence dataset from an NPZ archive."""
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        if not {"X", "y"}.issubset(archive.files):
            raise ValueError(f"{path} must contain arrays named X and y")
        features = np.asarray(archive["X"], dtype=np.float32)
        labels = np.asarray(archive["y"])

    if features.ndim != 3 or features.shape[1:] != (sequence_length, FEATURE_COUNT):
        raise ValueError(
            f"{path} X must have shape (N, {sequence_length}, {FEATURE_COUNT}); "
            f"got {features.shape}"
        )
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.reshape(-1)
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError(f"{path} y must contain one label per sequence")
    if not np.isfinite(features).all():
        raise ValueError(f"{path} X contains non-finite feature values")
    if not np.isin(labels, CLASS_LABELS).all():
        raise ValueError(f"{path} labels must be binary values 0 or 1")
    if features.shape[0] == 0:
        raise ValueError(f"{path} contains no sequences")

    return features, labels.astype(np.int64, copy=False)


def train_predictor() -> Path:
    """Train on prepared NPZ sequences and report test metrics for class 1.

    Class weights are computed only from training labels. The held-out test
    data is used to report precision, recall, and F1 for the congested class.

    Returns:
        Path to the saved model weights.
    """
    sequence_length = load_sequence_length()
    train_features, train_labels = load_dataset(TRAIN_PATH, sequence_length)
    test_features, test_labels = load_dataset(TEST_PATH, sequence_length)

    present_classes = np.unique(train_labels)
    if not np.array_equal(present_classes, CLASS_LABELS):
        raise ValueError(
            "Training data must contain both labels 0 and 1 to compute "
            f"class weights; found {present_classes.tolist()}"
        )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=CLASS_LABELS,
        y=train_labels,
    )
    class_counts = np.bincount(train_labels, minlength=CLASS_COUNT)
    print(f"Training label counts: {class_counts.tolist()}")
    print(f"Class weights [0, 1]: {class_weights.tolist()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrafficLSTM(input_size=FEATURE_COUNT).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters())
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features),
            torch.from_numpy(train_labels),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    for epoch in range(EPOCH_COUNT):
        model.train()
        epoch_loss = 0.0
        sample_count = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            batch_size = batch_labels.size(0)
            epoch_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        print(
            f"Epoch {epoch + 1}/{EPOCH_COUNT} - "
            f"train_loss: {epoch_loss / sample_count:.4f}"
        )

    model.eval()
    test_inputs = torch.from_numpy(test_features).to(device)
    with torch.no_grad():
        predicted_labels = model(test_inputs).argmax(dim=1).cpu().numpy()

    print("Test metrics for label 1 (Đông đúc):")
    print(
        classification_report(
            test_labels,
            predicted_labels,
            labels=[1],
            target_names=["Đông đúc"],
            zero_division=0,
            digits=4,
        )
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model weights to {MODEL_PATH}")
    return MODEL_PATH


if __name__ == "__main__":
    train_predictor()
