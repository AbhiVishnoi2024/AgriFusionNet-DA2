from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_tensor

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.agrifusionnet import AgriFusionNet
from src.models.baseline import EfficientNetBaseline

st.set_page_config(page_title="AgriFusionNet", layout="wide")

st.markdown(
    """
    <style>
    :root {
        --ink: #17211b;
        --muted: #617067;
        --line: #dfe7e1;
        --green: #1f6b4d;
        --green-soft: #edf6f0;
        --cream: #f7f9f5;
        --amber: #b97824;
    }
    .stApp {
        background: var(--cream);
        color: var(--ink);
    }
    [data-testid="stHeader"] {
        background: rgba(247, 249, 245, 0.88);
    }
    [data-testid="stSidebar"] {
        background: #eef4ef;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }
    .hero {
        border-bottom: 1px solid var(--line);
        padding: 1.2rem 0 1.1rem;
        margin-bottom: 1.5rem;
    }
    .eyebrow {
        color: var(--green);
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .hero h1 {
        color: var(--ink);
        font-size: 2.35rem;
        line-height: 1.05;
        margin: 0;
    }
    .hero p {
        color: var(--muted);
        font-size: 1rem;
        margin: 0.6rem 0 0;
    }
    .section-label {
        color: var(--ink);
        font-size: 1.15rem;
        font-weight: 700;
        margin: 1.3rem 0 0.7rem;
    }
    .upload-note {
        background: var(--green-soft);
        border: 1px solid #cfe3d5;
        border-left: 4px solid var(--green);
        color: #345345;
        padding: 0.75rem 0.9rem;
        margin: 0.7rem 0 1rem;
    }
    .result-strip {
        background: white;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.85rem 1rem;
        margin: 0.5rem 0 1.25rem;
    }
    .result-strip .label {
        color: var(--muted);
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .result-strip .value {
        color: var(--green);
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    .sidebar-title {
        color: var(--green);
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .sidebar-copy {
        color: var(--muted);
        font-size: 0.86rem;
        line-height: 1.45;
        margin-bottom: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    if ckpt["model_name"] == "baseline":
        model = EfficientNetBaseline(pretrained=False)
    else:
        model = AgriFusionNet(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), device, ckpt["model_name"], ckpt.get("img_size", 224)


def prepare(image: Image.Image, size: int):
    tensor = to_tensor(image.convert("RGB"))
    tensor = resize(tensor, [size, size])
    tensor = normalize(tensor, MEAN, STD)
    return tensor.unsqueeze(0)


def get_prediction(model, device, model_name, image, size):
    x = prepare(image, size).to(device)
    if model_name == "agrifusion":
        logits, aux = model(x, return_features=True)
    else:
        logits = model(x)
        aux = None
    probs = torch.softmax(logits, dim=1)[0]
    pred = int(probs.argmax())
    cam = None
    severity = None
    if model_name == "agrifusion":
        score = logits[:, pred].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        feats = aux["cnn_features"]
        grads = feats.grad
        weights = grads.mean(dim=(2, 3), keepdim=True)
        heat = (weights * feats).sum(dim=1, keepdim=True)
        heat = F.relu(heat)
        heat = F.interpolate(heat, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        cam = heat.detach().cpu().numpy()
        severity = float((cam >= 0.5).mean() * 100.0)
    return pred, probs.detach().cpu().numpy(), cam, severity


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">UAV crop health analysis · DA2 prototype</div>
        <h1>AgriFusionNet</h1>
        <p>Classify potato image patches for Alternaria symptoms and inspect the model's visual explanation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="sidebar-title">Model setup</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-copy">Choose a trained checkpoint. Relative paths are resolved from the project root.</div>',
    unsafe_allow_html=True,
)
checkpoint_input = st.sidebar.text_input("Model checkpoint", "weights/agrifusion_best.pth")
checkpoint_path = Path(checkpoint_input).expanduser()
if not checkpoint_path.is_absolute():
    checkpoint_path = PROJECT_ROOT / checkpoint_path
checkpoint_path = checkpoint_path.resolve()
st.sidebar.caption(f"Resolved path: {checkpoint_path}")

if not checkpoint_path.exists():
    st.error(f"Model checkpoint not found: {checkpoint_path}")
    st.info("Choose an existing checkpoint using the sidebar field.")
    st.stop()

model, device, model_name, img_size = load_model(checkpoint_path)
st.sidebar.divider()
st.sidebar.caption(f"Device: {device}")
st.sidebar.caption(f"Model: {model_name}")
st.sidebar.caption(f"Input size: {img_size} × {img_size}")

st.markdown('<div class="section-label">Upload an image patch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="upload-note">Use a clear RGB UAV patch in JPG, JPEG, or PNG format.</div>',
    unsafe_allow_html=True,
)
uploaded = st.file_uploader(
    "Choose image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    pred, probs, cam, severity = get_prediction(model, device, model_name, image, img_size)
    label = "Alternaria" if pred == 1 else "No Alternaria"

    st.markdown('<div class="section-label">Prediction</div>', unsafe_allow_html=True)
    result_col, confidence_col, status_col = st.columns([1.1, 1, 1.2])
    with result_col:
        st.markdown(
            f'<div class="result-strip"><div class="label">Predicted class</div><div class="value">{label}</div></div>',
            unsafe_allow_html=True,
        )
    with confidence_col:
        st.markdown(
            f'<div class="result-strip"><div class="label">Confidence</div><div class="value">{probs[pred] * 100:.2f}%</div></div>',
            unsafe_allow_html=True,
        )
    with status_col:
        st.markdown(
            '<div class="result-strip"><div class="label">Analysis status</div><div class="value">Complete</div></div>',
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown('<div class="section-label">Input image</div>', unsafe_allow_html=True)
        st.image(image, width="stretch")
    with c2:
        st.markdown('<div class="section-label">Class probabilities</div>', unsafe_allow_html=True)
        st.progress(float(probs[0]), text=f"No Alternaria · {probs[0] * 100:.2f}%")
        st.progress(float(probs[1]), text=f"Alternaria · {probs[1] * 100:.2f}%")

    if cam is not None:
        st.markdown('<div class="section-label">Explainability</div>', unsafe_allow_html=True)
        heat_rgba = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
        heat_img = Image.fromarray(heat_rgba).resize(image.size)
        overlay = Image.blend(image, heat_img, 0.45)
        explain_col, note_col = st.columns([1, 1], gap="large")
        with explain_col:
            st.image(
                overlay,
                caption="Grad-CAM: model explanation, not a ground-truth segmentation mask",
                width="stretch",
            )
        with note_col:
            st.markdown('<div class="section-label">Activation summary</div>', unsafe_allow_html=True)
            st.metric("Activation-based area estimate", f"{severity:.2f}%")
            st.info(
                "This percentage is calculated from the activation heatmap threshold. "
                "It is not a pixel-level disease measurement."
            )
