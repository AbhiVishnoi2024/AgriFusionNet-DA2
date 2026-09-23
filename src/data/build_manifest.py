from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm(s: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def infer_label_from_text(text: str) -> Optional[int]:
    n = norm(text)
    if any(x in n for x in ["no alternaria", "not alternaria", "non alternaria", "no_alternaria", "0 no alternaria"]):
        return 0
    if "alternaria" in n:
        return 1
    if re.search(r"(^|[ _-])1([ _-]|$)", n):
        return 1
    if re.search(r"(^|[ _-])0([ _-]|$)", n):
        return 0
    return None


def find_csv_rows(root: Path) -> pd.DataFrame:
    rows = []
    filename_priority = [
        "Original_name", "Original patch name", "Original image name", "Random_name",
        "Random patch name", "random patch name", "Patch name", "patch name",
        "file", "filename", "image", "name"
    ]

    def first_value(row: pd.Series, names: list[str]):
        for name in names:
            if name in row.index and pd.notna(row[name]):
                return row[name]
        return None

    for csv_path in root.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        label_cols = [c for c in df.columns if "label" in norm(c)]
        if not label_cols:
            continue
        label_col = label_cols[0]
        file_cols = [c for c in filename_priority if c in df.columns]
        if not file_cols:
            file_cols = [c for c in df.columns if any(k in norm(c) for k in ["patch name", "file", "image", "name"])]
        if not file_cols:
            continue
        for _, r in df.iterrows():
            raw_label = r[label_col]
            try:
                label = int(raw_label)
            except Exception:
                label = infer_label_from_text(str(raw_label))
            if label not in (0, 1):
                continue
            name = first_value(r, file_cols)
            if name is None:
                continue
            original_name = first_value(r, [
                "Original_name", "Original image name", "Original patch name", "Block"
            ]) or name
            rows.append({
                "filename": Path(str(name)).name,
                "csv_path": str(csv_path),
                "csv_label": label,
                "original_name": str(original_name),
                "latitude": first_value(r, ["Lat", "Latitude", "latitude"]),
                "longitude": first_value(r, ["Lon", "Longitude", "longitude"]),
                "altitude": first_value(r, ["Alt", "Altitude", "altitude"]),
                "date": first_value(r, ["Date", "Date of flight", "date"]),
            })
    return pd.DataFrame(rows)


def build(root: Path, output: Path) -> None:
    csv_rows = find_csv_rows(root)
    lookup = {}
    if not csv_rows.empty:
        for _, row in csv_rows.iterrows():
            lookup[row["filename"]] = row.to_dict()

    records = []
    for image_path in sorted(root.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        csv_info = lookup.get(image_path.name, {})
        label = csv_info.get("csv_label")
        if pd.isna(label) if label is not None else True:
            label = infer_label_from_text(str(image_path.parent))
        if label not in (0, 1):
            label = infer_label_from_text(image_path.name)
        if label not in (0, 1):
            continue
        year = None
        for part in image_path.parts:
            if part in {"2019", "2022"}:
                year = int(part)
                break
        if year is None:
            match = re.search(r"20(19|22)", str(image_path))
            year = int(match.group(0)) if match else 0
        records.append({
            "path": str(image_path.resolve()),
            "label": int(label),
            "class_name": "Alternaria" if int(label) == 1 else "No Alternaria",
            "year": year,
            "group_id": str(csv_info.get("original_name", image_path.stem)),
            "latitude": csv_info.get("latitude"),
            "longitude": csv_info.get("longitude"),
            "altitude": csv_info.get("altitude"),
            "date": csv_info.get("date"),
        })

    if not records:
        raise RuntimeError(
            "No labeled images found. Extract the dataset under data/raw/2019 and data/raw/2022. "
            "The script uses the provided CSV labels when available and folder names as a fallback."
        )

    df = pd.DataFrame(records).drop_duplicates(subset=["path"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Wrote {len(df)} images to {output}")
    print(df.groupby(["year", "class_name"]).size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw")
    parser.add_argument("--output", default="data/processed/manifest.csv")
    args = parser.parse_args()
    build(Path(args.root), Path(args.output))
