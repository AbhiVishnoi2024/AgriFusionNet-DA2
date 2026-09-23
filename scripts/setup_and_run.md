# DA2 setup and run checklist

## 1. Create the environment

Windows PowerShell or Git Bash:

```bash
cd /d D:\AgriFusionNet_DA2
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
# Git Bash
source .venv/Scripts/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For training, Google Colab with a GPU is recommended. Upload the repository to Colab and run the same Python commands from the repository root.

## 2. Download the dataset

Use the official Zenodo record:
https://zenodo.org/records/10727413

Download:
- `Alternaria_2019.zip`
- `Alternaria_2022.zip`

Extract them to:

```text
data/raw/2019/
data/raw/2022/
```

Do not modify the original labels.

## 3. Build the manifest

```bash
python -m src.data.build_manifest --root data/raw --output data/processed/manifest.csv
```

Check that you get counts for both 2019 and 2022 and both classes.

## 4. Make the cross-season split

```bash
python -m src.data.split_manifest --manifest data/processed/manifest.csv --output-dir data/processed/splits
```

Planned split:
- Train: 2019
- Validation: 15% of 2019
- Test: all 2022

This keeps the 2022 season independent for the final test.

## 5. Smoke test the model code

```bash
python -m src.models.agrifusionnet
```

Expected: a 2-class output tensor.

## 6. Train a baseline first

```bash
python -m src.train --model baseline --epochs 5 --batch-size 16 --output weights/baseline_best.pth
```

Then evaluate:

```bash
python -m src.evaluate --checkpoint weights/baseline_best.pth --output-dir results/baseline
```

## 7. Train AgriFusionNet

```bash
python -m src.train --model agrifusion --epochs 5 --batch-size 8 --output weights/agrifusion_best.pth
```

Then:

```bash
python -m src.evaluate --checkpoint weights/agrifusion_best.pth --output-dir results/agrifusion
```

Start with frozen ImageNet backbones. Only unfreeze later if compute allows:

```bash
python -m src.train --model agrifusion --epochs 5 --batch-size 4 --unfreeze-backbone --output weights/agrifusion_finetuned.pth
```

## 8. Run the dashboard

From the repository root:

```bash
streamlit run dashboard/app.py
```

Set the checkpoint path in the sidebar, upload a UAV image patch, and show the disease prediction.

## 9. Generate a terminal prediction

```bash
python -m src.predict --checkpoint weights/agrifusion_best.pth --image path\to\sample.jpg
```

## 10. What to record for the review

Take screenshots of:
1. Dataset folder and sample images
2. Manifest class/year counts
3. Architecture diagram
4. Training output
5. Accuracy / precision / recall / F1
6. Confusion matrix
7. Dashboard prediction
8. Grad-CAM localization
9. GitHub repository

Never invent a metric. Copy the values printed by `evaluate.py` into the PPT/report.
