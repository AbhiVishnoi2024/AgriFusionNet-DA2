# 🏗️ AgriFusionNet — Complete Architecture

```mermaid
flowchart TD

    %% =========================
    %% DATA PIPELINE
    %% =========================

    A["🌱 UAV RGB Dataset<br/>7,660 Labeled Patches<br/>256 × 256 RGB"] 
        --> B["🔍 Dataset Verification<br/>verify_dataset.py"]

    B --> C["📋 Manifest Creation<br/>build_manifest.py"]

    C --> D["manifest.csv<br/>Image Path + Label + Metadata"]

    D --> E["✂️ Dataset Splitting<br/>split_manifest.py"]

    E --> F["Training Set"]
    E --> G["Validation Set"]
    E --> H["Independent 2022 Test Set"]

    %% =========================
    %% PREPROCESSING
    %% =========================

    F --> I["⚙️ dataset.py<br/>Image Loading + Transformation"]
    G --> I

    I --> J["🖼️ Preprocessing<br/>RGB → Tensor<br/>Resize → 224 × 224<br/>ImageNet Normalization"]

    %% =========================
    %% AGRIFUSIONNET
    %% =========================

    J --> K["🧠 AgriFusionNet"]

    K --> L["EfficientNet-B3<br/>CNN Branch"]
    K --> M["Swin-Tiny<br/>Transformer Branch"]

    %% CNN
    L --> L1["1536-Channel<br/>Feature Map"]
    L1 --> L2["1 × 1 Projection<br/>1536 → 256"]

    %% Transformer
    M --> M1["768-Channel<br/>Feature Map"]
    M1 --> M2["1 × 1 Projection<br/>768 → 256"]

    %% Pooling
    L2 --> N["Adaptive Average Pooling<br/>7 × 7"]
    M2 --> O["Adaptive Average Pooling<br/>7 × 7"]

    %% Tokens
    N --> P["CNN Tokens<br/>49 × 256"]
    O --> Q["Transformer Tokens<br/>49 × 256"]

    %% Cross Attention
    P --> R["🔗 Bidirectional<br/>Cross-Attention"]
    Q --> R

    R --> R1["CNN → Transformer<br/>Attention"]
    R --> R2["Transformer → CNN<br/>Attention"]

    R1 --> S["Feature Fusion"]
    R2 --> S

    S --> T["Feed-Forward Network<br/>+ Residual Refinement"]

    T --> U["Mean Pooling<br/>49 Tokens → 256-D Vector"]

    %% Classification
    U --> V["🎯 Classification Head<br/>LayerNorm<br/>Linear 256 → 128<br/>GELU<br/>Dropout 0.25<br/>Linear 128 → 2"]

    V --> W["Softmax"]

    W --> X["0 — No Alternaria"]
    W --> Y["1 — Alternaria"]

    %% =========================
    %% GRAD-CAM
    %% =========================

    V --> Z["🔥 Grad-CAM"]
    Z --> ZA["Explainability Heatmap"]
    ZA --> ZB["Highlights Image Regions<br/>Influencing Prediction"]

    %% =========================
    %% EVALUATION
    %% =========================

    H --> AA["📊 Independent Test Evaluation"]

    AA --> AB["Accuracy"]
    AA --> AC["Precision"]
    AA --> AD["Recall"]
    AA --> AE["F1 / Macro-F1"]

    %% =========================
    %% DASHBOARD
    %% =========================

    X --> AF["🖥️ Streamlit Dashboard"]
    Y --> AF
    ZA --> AF

    AF --> AG["Prediction"]
    AF --> AH["Confidence"]
    AF --> AI["Class Probabilities"]
    AF --> AJ["Grad-CAM Visualization"]

    %% =========================
    %% BASELINE
    %% =========================

    D --> AK["🆚 Baseline Experiment"]

    AK --> AL["EfficientNet-B0"]
    AL --> AM["Classification Head"]
    AM --> AN["Alternaria / No Alternaria"]

    AN --> AA
