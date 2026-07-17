"""
Synthetic dashboard warning-light image dataset.

In a production setting this would be replaced with real dashboard photos
(BMW's own vehicles could source this from service-center intake photos or
owner app uploads). For a self-contained, offline-runnable portfolio project,
we procedurally generate labeled icon images with randomized jitter (position,
rotation, colour noise) — this keeps the classification task genuinely
non-trivial (a model that just memorizes pixel positions will not generalize
to the held-out validation split) while requiring zero external data
download.

Classes mirror the five warning lights in app/tools.py's WARNING_LIGHTS table,
so the CV tool and the text-lookup tool stay consistent with each other.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw

IMAGE_SIZE = 32
CLASSES = ["engine", "oil", "battery", "brake", "tire"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

_COLORS = {
    "engine": (255, 176, 0),   # amber
    "oil": (220, 20, 20),      # red
    "battery": (220, 20, 20),  # red
    "brake": (220, 20, 20),    # red
    "tire": (255, 176, 0),     # amber
}


def _draw_icon(label: str, jitter: bool = True) -> Image.Image:
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (10, 10, 10))
    draw = ImageDraw.Draw(img)
    color = _COLORS[label]

    cx, cy = IMAGE_SIZE // 2, IMAGE_SIZE // 2
    if jitter:
        cx += random.randint(-3, 3)
        cy += random.randint(-3, 3)
    r = random.randint(9, 12) if jitter else 10

    if label == "engine":
        # amber circle with a small rectangle "block" on top
        draw.ellipse((cx - r, cy - r + 3, cx + r, cy + r + 3), outline=color, width=2)
        draw.rectangle((cx - 4, cy - r - 4, cx + 4, cy - r + 3), fill=color)
    elif label == "oil":
        # red teardrop-ish shape: circle + triangle on top
        draw.ellipse((cx - r, cy - r + 4, cx + r, cy + r + 4), fill=color)
        draw.polygon([(cx, cy - r - 6), (cx - 6, cy - r + 4), (cx + 6, cy - r + 4)], fill=color)
    elif label == "battery":
        # red rectangle with two terminal nubs
        draw.rectangle((cx - r, cy - r + 4, cx + r, cy + r - 2), outline=color, width=2)
        draw.rectangle((cx - 4, cy - r - 3, cx - 1, cy - r + 4), fill=color)
        draw.rectangle((cx + 1, cy - r - 3, cx + 4, cy - r + 4), fill=color)
    elif label == "brake":
        # red circle with exclamation mark
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=2)
        draw.line((cx, cy - 5, cx, cy + 2), fill=color, width=2)
        draw.ellipse((cx - 1, cy + 5, cx + 1, cy + 7), fill=color)
    elif label == "tire":
        # amber horseshoe arc with exclamation
        draw.arc((cx - r, cy - r, cx + r, cy + r), start=30, end=330, fill=color, width=3)
        draw.line((cx, cy - 4, cx, cy + 2), fill=color, width=2)
        draw.ellipse((cx - 1, cy + 4, cx + 1, cy + 6), fill=color)

    if jitter:
        # mild colour/pixel noise so the task isn't trivially separable by exact pixels
        arr = np.array(img).astype(np.int16)
        noise = np.random.randint(-15, 15, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


def generate_dataset(n_per_class: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Returns (images, labels) as numpy arrays: images shape (N, 32, 32, 3) uint8, labels shape (N,)."""
    random.seed(seed)
    np.random.seed(seed)

    images, labels = [], []
    for label in CLASSES:
        for _ in range(n_per_class):
            img = _draw_icon(label, jitter=True)
            images.append(np.array(img))
            labels.append(CLASS_TO_IDX[label])

    images = np.stack(images).astype(np.uint8)
    labels = np.array(labels, dtype=np.int64)

    # shuffle
    perm = np.random.permutation(len(labels))
    return images[perm], labels[perm]


def save_sample_image(label: str, path: str) -> None:
    """Utility used by tests/demos to produce a single sample PNG for a given class."""
    img = _draw_icon(label, jitter=True)
    img.save(path)
