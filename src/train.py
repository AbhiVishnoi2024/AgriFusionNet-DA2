from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import UAVPatchDataset
from src.models.agrifusionnet import AgriFusionNet
from src.models.baseline import EfficientNetBaseline
from src.utils.metrics import classification_metrics, save_json


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(name: str, pretrained: bool, freeze: bool):
    if name == "baseline":
        return EfficientNetBaseline(pretrained=pretrained, freeze_backbone=freeze)
    return AgriFusionNet(pretrained=pretrained, freeze_backbones=freeze)


def run_epoch(model, loader, criterion, optimizer, scaler, device, training=True):
    model.train(training)
    running_loss = 0.0
    all_y, all_pred = [], []
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)
        if training:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        running_loss += loss.item() * images.size(0)
        all_y.extend(labels.detach().cpu().tolist())
        all_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    metrics = classification_metrics(all_y, all_pred)
    metrics["loss"] = running_loss / max(len(loader.dataset), 1)
    return metrics


def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    train_df = pd.read_csv(train_csv)
    counts = train_df["label"].value_counts().to_dict()
    weight0 = len(train_df) / (2 * counts.get(0, 1))
    weight1 = len(train_df) / (2 * counts.get(1, 1))
    class_weights = torch.tensor([weight0, weight1], dtype=torch.float32, device=device)
    print("Class weights:", class_weights.detach().cpu().tolist())

    train_ds = UAVPatchDataset(train_csv, args.img_size, train=True)
    val_ds = UAVPatchDataset(val_csv, args.img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = make_model(args.model, pretrained=not args.no_pretrained, freeze=not args.unfreeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(args.scheduler_patience, 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True)
        val_m = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()}, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(row)
        print(row)
        scheduler.step(val_m["f1"])
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_name": args.model,
                "img_size": args.img_size,
                "epoch": epoch,
                "best_val_f1": best_f1,
                "class_mapping": {"0": "No Alternaria", "1": "Alternaria"},
                "config": vars(args),
                "class_weights": class_weights.detach().cpu().tolist(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }, out)
            print("Saved best model ->", out)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {epoch} epochs; best epoch was {best_epoch}.")
                break

    save_json({
        "device": str(device),
        "best_val_f1": best_f1,
        "best_epoch": best_epoch,
        "history": history,
    }, out.with_suffix(".json"))
    if history:
        with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "agrifusion"], default="baseline")
    parser.add_argument("--train-csv", default="data/processed/splits/train.csv")
    parser.add_argument("--val-csv", default="data/processed/splits/val.csv")
    parser.add_argument("--output", default="weights/best_model.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--scheduler-patience", type=int, default=1)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    args = parser.parse_args()
    main(args)
