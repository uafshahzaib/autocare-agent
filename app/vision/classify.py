"""Inference-time wrapper around the trained WarningLightCNN."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from app.vision.dataset import CLASSES, IMAGE_SIZE
from app.vision.model import WarningLightCNN
from app.vision.train import MODEL_PATH


@lru_cache(maxsize=1)
def _load_model() -> WarningLightCNN:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run `python -m app.vision.train` first."
        )
    model = WarningLightCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def classify_image(image_path: str) -> tuple[str, float]:
    """Returns (predicted_class_label, confidence) for a dashboard warning-light image."""
    model = _load_model()

    img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 32, 32)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        idx = int(probs.argmax())

    return CLASSES[idx], float(probs[idx])
