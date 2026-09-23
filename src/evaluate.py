from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import UAVPatchDataset
from src.models.agrifusionnet import AgriFusionNet
from src.models.baseline import EfficientNetBaseline
from src.utils.metrics import (
    classification_metrics,
    classification_report_text,
    save_confusion_matrix,
    save_json,
)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt["model_name"]
    img_size = ckpt.get("img_size", args.img_size)
    model = EfficientNetBaseline(pretrained=False) if model_name == "baseline" else AgriFusionNet(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    ds = UAVPatchDataset(args.test_csv, img_size, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            logits = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = classification_metrics(y_true, y_pred)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out / "metrics.json")
    (out / "classification_report.txt").write_text(
        classification_report_text(y_true, y_pred), encoding="utf-8"
    )
    save_confusion_matrix(y_true, y_pred, out / "confusion_matrix.png")
    print(metrics)
    print("Results written to", out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/best_model.pth")
    parser.add_argument("--test-csv", default="data/processed/splits/test.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    main(args)
