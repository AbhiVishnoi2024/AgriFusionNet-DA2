from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from src.data.build_manifest import find_csv_rows
from src.data.dataset import UAVPatchDataset
from src.models.agrifusionnet import AgriFusionNet


class DataAndModelTests(unittest.TestCase):
    def test_metadata_parser_reads_dataset_column_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame([{
                "Original_name": "sample.JPG",
                "Lat": 50.0,
                "Lon": 3.0,
                "Alt": 20.0,
                "Date": 20220704,
                "Label": 1,
            }])
            frame.to_csv(root / "metadata.csv", index=False)
            rows = find_csv_rows(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows.iloc[0]["filename"], "sample.JPG")
            self.assertEqual(rows.iloc[0]["csv_label"], 1)
            self.assertEqual(rows.iloc[0]["latitude"], 50.0)

    def test_dataset_returns_image_label_and_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "sample.jpg"
            Image.new("RGB", (32, 32), (20, 100, 40)).save(image_path)
            csv_path = root / "split.csv"
            pd.DataFrame([{"path": str(image_path), "label": 1}]).to_csv(csv_path, index=False)
            image, label, path = UAVPatchDataset(csv_path, img_size=16)[0]
            self.assertEqual(tuple(image.shape), (3, 16, 16))
            self.assertEqual(int(label), 1)
            self.assertEqual(Path(path), image_path)

    def test_agrifusionnet_outputs_two_logits(self):
        model = AgriFusionNet(pretrained=False, freeze_backbones=True)
        with torch.no_grad():
            output = model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()