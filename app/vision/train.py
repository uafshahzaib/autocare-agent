"""
Trains WarningLightCNN on the synthetic dataset and saves weights to
app/vision/model.pt. Run once (or whenever the dataset generator changes):

    python -m app.vision.train
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.vision.dataset import generate_dataset
from app.vision.model import WarningLightCNN

MODEL_PATH = Path(__file__).parent / "model.pt"


def _to_tensor(images: np.ndarray) -> torch.Tensor:
    # (N, H, W, C) uint8 -> (N, C, H, W) float32 normalized to [0, 1]
    t = torch.from_numpy(images).float() / 255.0
    return t.permute(0, 3, 1, 2)


def train(epochs: int = 15, batch_size: int = 32, lr: float = 1e-3, seed: int = 42) -> float:
    images, labels = generate_dataset(n_per_class=200, seed=seed)
    val_images, val_labels = generate_dataset(n_per_class=40, seed=seed + 1)

    train_ds = TensorDataset(_to_tensor(images), torch.from_numpy(labels))
    val_ds = TensorDataset(_to_tensor(val_images), torch.from_numpy(val_labels))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = WarningLightCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        train_loss = total_loss / len(train_ds)
        val_acc = _evaluate(model, val_loader)
        print(f"epoch {epoch + 1:2d}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.1%}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")
    return val_acc


def _evaluate(model: WarningLightCNN, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total


if __name__ == "__main__":
    train()
