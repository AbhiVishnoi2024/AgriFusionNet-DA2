from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def split(df: pd.DataFrame, output_dir: Path, val_size: float = 0.15, seed: int = 42) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    trainval = df[df["year"] == 2019].copy()
    test = df[df["year"] == 2022].copy()
    if trainval.empty or test.empty:
        raise RuntimeError("Both 2019 and 2022 data are required for the planned cross-season split.")

    # Stratified patch-level split inside 2019. The independent 2022 season remains untouched for testing.
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(trainval, trainval["label"]))
    train = trainval.iloc[train_idx].copy()
    val = trainval.iloc[val_idx].copy()

    train.to_csv(output_dir / "train.csv", index=False)
    val.to_csv(output_dir / "val.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)

    print("TRAIN", len(train), train["class_name"].value_counts().to_dict())
    print("VAL  ", len(val), val["class_name"].value_counts().to_dict())
    print("TEST ", len(test), test["class_name"].value_counts().to_dict())
    print(f"Saved splits to {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--output-dir", default="data/processed/splits")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split(pd.read_csv(args.manifest), Path(args.output_dir), args.val_size, args.seed)
