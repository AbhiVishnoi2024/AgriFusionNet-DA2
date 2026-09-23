from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_LABELS = {0, 1}


def verify_dataset(root: Path, remove_invalid: bool = False) -> int:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    image_files = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    unsupported_images = [
        path for path in files
        if path.suffix and path.suffix.lower() not in IMAGE_EXTENSIONS
        and path.suffix.lower() in {".tif", ".tiff", ".gif"}
    ]

    invalid_images: list[tuple[Path, str]] = []
    dimensions: Counter[tuple[int, int]] = Counter()
    for path in image_files:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions[image.size] += 1
        except Exception as exc:
            invalid_images.append((path, str(exc)))

    csv_files = sorted(root.rglob("*.csv"))
    metadata_rows = 0
    invalid_labels: list[tuple[Path, int, object]] = []
    missing_label_rows = 0
    for csv_path in csv_files:
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Unreadable CSV: {csv_path} ({exc})")
            continue
        metadata_rows += len(frame)
        label_columns = [column for column in frame.columns if str(column).strip().lower() == "label"]
        if not label_columns:
            missing_label_rows += len(frame)
            continue
        for row_number, value in frame[label_columns[0]].items():
            try:
                label = int(value)
            except (TypeError, ValueError):
                label = None
            if label not in VALID_LABELS:
                invalid_labels.append((csv_path, int(row_number) + 2, value))

    paths = [str(path.resolve()) for path in image_files]
    duplicate_paths = len(paths) - len(set(paths))
    class_counts = Counter()
    for path in image_files:
        parts = {part.lower() for part in path.parts}
        if "1" in parts:
            class_counts["Alternaria"] += 1
        elif "0" in parts:
            class_counts["No Alternaria"] += 1
        else:
            class_counts["unresolved"] += 1

    print(f"Dataset root: {root.resolve()}")
    print(f"Files: {len(files)}")
    print(f"Images: {len(image_files)}")
    print(f"Metadata CSV files: {len(csv_files)} ({metadata_rows} rows)")
    print(f"Class distribution by folder: {dict(sorted(class_counts.items()))}")
    print(f"Image dimensions: {dict(dimensions)}")
    print(f"Unreadable/corrupt images: {len(invalid_images)}")
    print(f"Duplicate image paths: {duplicate_paths}")
    print(f"Unsupported TIFF/GIF images: {len(unsupported_images)}")
    print(f"Missing-label metadata rows: {missing_label_rows}")
    print(f"Invalid-label metadata rows: {len(invalid_labels)}")

    if invalid_images:
        print("Invalid image files:")
        for path, error in invalid_images:
            print(f"  {path}: {error}")
        if remove_invalid:
            for path, _ in invalid_images:
                path.unlink()
            print(f"Removed {len(invalid_images)} invalid image files.")

    if invalid_labels:
        print("Invalid metadata labels:")
        for csv_path, row_number, value in invalid_labels[:20]:
            print(f"  {csv_path}:{row_number} -> {value!r}")

    return 1 if invalid_images or duplicate_paths or invalid_labels or missing_label_rows else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify UAV image and label data without deleting files by default.")
    parser.add_argument("--root", default="data/raw", type=Path)
    parser.add_argument("--remove-invalid", action="store_true")
    args = parser.parse_args()
    raise SystemExit(verify_dataset(args.root, args.remove_invalid))