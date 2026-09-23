from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_tensor

from src.models.agrifusionnet import AgriFusionNet
from src.models.baseline import EfficientNetBaseline

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    model_name = ckpt["model_name"]
    model = EfficientNetBaseline(pretrained=False) if model_name == "baseline" else AgriFusionNet(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), model_name, ckpt.get("img_size", 224)


def make_input(path: str, img_size: int):
    image = Image.open(path).convert("RGB")
    t = to_tensor(image)
    t = resize(t, [img_size, img_size])
    t = normalize(t, MEAN, STD)
    return image, t.unsqueeze(0)


def gradcam(model, x, pred_class):
    if not isinstance(model, AgriFusionNet):
        return None
    logits, aux = model(x, return_features=True)
    score = logits[:, pred_class].sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    feats = aux["cnn_features"]
    grads = feats.grad
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * feats).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(x.shape[-2], x.shape[-1]), mode="bilinear", align_corners=False)
    cam = cam[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    model, model_name, img_size = load_model(args.checkpoint, device)
    load_time = time.perf_counter() - start
    preprocess_start = time.perf_counter()
    original, x = make_input(args.image, img_size)
    preprocess_time = time.perf_counter() - preprocess_start
    x = x.to(device)
    inference_start = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    inference_time = time.perf_counter() - inference_start
    pred = int(probs.argmax())
    print({"model": model_name, "prediction": "Alternaria" if pred == 1 else "No Alternaria", "confidence": float(probs[pred])})
    print({
        "preprocessing_seconds": preprocess_time,
        "inference_seconds": inference_time,
        "model_load_seconds": load_time,
        "total_seconds": time.perf_counter() - start,
    })

    if model_name == "agrifusion":
        cam = gradcam(model, x.requires_grad_(True), pred)
        severity = float((cam >= args.threshold).mean() * 100.0)
        print(f"Activation-based localization area estimate: {severity:.2f}%")
        np.save(args.output, cam)
        print("Saved heatmap array to", args.output)
        if args.output_image:
            import matplotlib.cm as cm

            heat = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
            heat_image = Image.fromarray(heat).resize(original.size)
            overlay = Image.blend(original, heat_image, 0.45)
            overlay.save(args.output_image)
            print("Saved Grad-CAM overlay to", args.output_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/best_model.pth")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="results/gradcam.npy")
    parser.add_argument("--output-image", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
