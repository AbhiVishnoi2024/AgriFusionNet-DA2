from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.models.baseline import EfficientNetBaseline
from src.predict import load_model, make_input


class InferenceTests(unittest.TestCase):
    def test_checkpoint_load_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pth"
            model = EfficientNetBaseline(pretrained=False)
            torch.save({
                "model_state": model.state_dict(),
                "model_name": "baseline",
                "img_size": 32,
            }, checkpoint)
            loaded, model_name, image_size = load_model(checkpoint, torch.device("cpu"))
            image = Image.new("RGB", (64, 64), (100, 120, 140))
            original, tensor = make_input_from_image(image, image_size)
            self.assertEqual(model_name, "baseline")
            self.assertEqual(image_size, 32)
            self.assertEqual(original.size, (64, 64))
            self.assertEqual(tuple(tensor.shape), (1, 3, 32, 32))
            with torch.no_grad():
                self.assertEqual(tuple(loaded(tensor).shape), (1, 2))


def make_input_from_image(image: Image.Image, image_size: int):
    handle = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    try:
        handle.close()
        image.save(handle.name)
        return make_input(handle.name, image_size)
    finally:
        Path(handle.name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()