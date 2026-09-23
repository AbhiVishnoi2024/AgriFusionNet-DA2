from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.cm as cm
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import UAVPatchDataset
from src.predict import gradcam, load_model, make_input

CLASS_NAMES = {0: "No Alternaria", 1: "Alternaria"}


def save_demo_image(image_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        image.convert("RGB").save(output_path, format="JPEG", quality=95)


def save_gradcam_overlay(model, image_path: Path, output_path: Path, image_size: int, pred_class: int) -> None:
    original, tensor = make_input(str(image_path), image_size)
    tensor = tensor.to(next(model.parameters()).device)
    cam = gradcam(model, tensor.requires_grad_(True), pred_class)
    if cam is None:
        raise RuntimeError("Grad-CAM is unavailable for the selected model.")

    heat = (cm.jet(cam)[:, :, :3] * 255).astype("uint8")
    heat_image = Image.fromarray(heat).resize(original.size)
    overlay = Image.blend(original, heat_image, 0.45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path, format="JPEG", quality=95)


def select_examples(
    checkpoint: Path,
    test_csv: Path,
    output_dir: Path,
    batch_size: int,
    workers: int,
    preferred_confidence: float,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name, image_size = load_model(checkpoint, device)
    if model_name != "agrifusion":
        raise ValueError(f"Checkpoint model is {model_name!r}; an AgriFusionNet checkpoint is required.")

    test_frame = pd.read_csv(test_csv)
    dataset = UAVPatchDataset(test_csv, img_size=image_size, train=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    candidates: dict[int, dict] = {}
    scanned = 0
    scan_start = time.perf_counter()
    for images, labels, paths in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device)
        inference_start = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(model(images), dim=1)
        batch_inference_time = time.perf_counter() - inference_start
        per_image_time = batch_inference_time / max(images.shape[0], 1)

        predictions = probabilities.argmax(dim=1)
        for index, (true_label, predicted_label, path) in enumerate(
            zip(labels.tolist(), predictions.tolist(), paths)
        ):
            scanned += 1
            confidence = float(probabilities[index, predicted_label].item())
            if true_label != predicted_label:
                continue
            current = candidates.get(true_label)
            if current is None or confidence > current["confidence"]:
                candidates[true_label] = {
                    "image_path": str(Path(path).resolve()),
                    "true_label": int(true_label),
                    "true_class": CLASS_NAMES[int(true_label)],
                    "predicted_label": int(predicted_label),
                    "predicted_class": CLASS_NAMES[int(predicted_label)],
                    "confidence": confidence,
                    "inference_time": per_image_time,
                }

        if all(label in candidates for label in (0, 1)) and all(
            candidates[label]["confidence"] >= preferred_confidence for label in (0, 1)
        ):
            break

    if set(candidates) != {0, 1}:
        missing = ", ".join(CLASS_NAMES[label] for label in (0, 1) if label not in candidates)
        raise RuntimeError(
            f"Could not find correctly classified examples for: {missing}. "
            f"Scanned {scanned} of {len(test_frame)} test images."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    selected = {
        "no_alternaria": candidates[0],
        "alternaria": candidates[1],
    }
    file_names = {
        "no_alternaria": ("no_alternaria_original.jpg", "no_alternaria_gradcam.jpg"),
        "alternaria": ("alternaria_original.jpg", "alternaria_gradcam.jpg"),
    }
    for key, record in selected.items():
        image_path = Path(record["image_path"])
        original_name, gradcam_name = file_names[key]
        save_demo_image(image_path, output_dir / original_name)
        save_gradcam_overlay(
            model,
            image_path,
            output_dir / gradcam_name,
            image_size,
            record["predicted_label"],
        )
        record["original_output"] = str((output_dir / original_name).resolve())
        record["gradcam_output"] = str((output_dir / gradcam_name).resolve())

    result = {
        "checkpoint": str(checkpoint.resolve()),
        "test_split": str(test_csv.resolve()),
        "test_set_description": "Independent 2022 test set; examples were selected without training on this split.",
        "gradcam_description": "Grad-CAM is an explainability/localization visualization, not ground-truth segmentation.",
        "device": str(device),
        "model": model_name,
        "images_scanned": scanned,
        "scan_seconds": time.perf_counter() - scan_start,
        "preferred_confidence": preferred_confidence,
        "examples": selected,
    }
    (output_dir / "demo_examples.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("Selected examples from the independent 2022 test set.")
    print("Grad-CAM is an explainability/localization visualization, not ground-truth segmentation.")
    for key, record in selected.items():
        print(
            f"{key}: {record['image_path']} | "
            f"true={record['true_class']} | predicted={record['predicted_class']} | "
            f"confidence={record['confidence']:.4f} | "
            f"inference_seconds={record['inference_time']:.6f}"
        )
    print(f"Wrote demo artifacts to {output_dir.resolve()}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select correct AgriFusionNet demo examples from the independent 2022 test set.")
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/agrifusion_best.pth"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/processed/splits/test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/demo"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--preferred-confidence",
        type=float,
        default=0.80,
        help="Stop once both classes have a correct example at or above this confidence; otherwise use the best observed examples.",
    )
    args = parser.parse_args()
    select_examples(
        args.checkpoint,
        args.test_csv,
        args.output_dir,
        args.batch_size,
        args.workers,
        args.preferred_confidence,
    )