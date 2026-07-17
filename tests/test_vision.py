"""
Tests for the CV component. These require torch/PIL to be installed but not
a trained model.pt (dataset generation and model forward-pass shape checks
don't need trained weights). The end-to-end classify_image() call is tested
separately and requires `python -m app.vision.train` to have been run first.
"""
import numpy as np
import pytest

from app.vision.dataset import CLASSES, IMAGE_SIZE, generate_dataset


def test_generate_dataset_shapes():
    images, labels = generate_dataset(n_per_class=5)
    assert images.shape == (5 * len(CLASSES), IMAGE_SIZE, IMAGE_SIZE, 3)
    assert labels.shape == (5 * len(CLASSES),)
    assert set(np.unique(labels).tolist()) == set(range(len(CLASSES)))


def test_model_forward_pass_shape():
    torch = pytest.importorskip("torch")
    from app.vision.model import WarningLightCNN

    model = WarningLightCNN()
    dummy = torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = model(dummy)
    assert out.shape == (2, len(CLASSES))


def test_classify_image_end_to_end(tmp_path):
    """Skips gracefully if the model hasn't been trained yet (expected in CI
    unless `python -m app.vision.train` is run as a setup step)."""
    from app.vision.dataset import save_sample_image
    from app.vision.train import MODEL_PATH

    if not MODEL_PATH.exists():
        pytest.skip("Run `python -m app.vision.train` before this test.")

    from app.vision.classify import classify_image

    sample_path = tmp_path / "sample_oil.png"
    save_sample_image("oil", str(sample_path))

    label, confidence = classify_image(str(sample_path))
    assert label in CLASSES
    assert 0.0 <= confidence <= 1.0
