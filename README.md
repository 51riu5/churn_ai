# Deep Learning–Driven Customer Churn Forecasting

A novel **Multi-Scale Temporal Attention Network (MSTAN)** for customer churn prediction on the IBM Telco dataset, featuring temporal decay attention, focal loss for class imbalance, and a comprehensive comparative study against four baseline architectures.

## Novelty — MSTAN Architecture

The core contribution is **MSTAN**, a purpose-built architecture for temporal churn prediction with three novel components:

1. **Multi-Scale Temporal Convolutions** — Parallel dilated causal convolutions (scales 1, 2, 4) with a **gated fusion** mechanism that learns to weight each temporal resolution per customer.
2. **Temporal Decay Attention Bias** — A learnable exponential decay added to attention logits so the model inherently attends more to recent customer interactions while still allowing long-range attention.
3. **Focal Loss** — Replaces standard BCE to down-weight easy/majority-class examples and focus training on hard-to-classify churners (alpha=0.75, gamma=2.0).

## Results

| Model | F1 | Recall | Precision | ROC-AUC | PR-AUC |
|-------|---:|-------:|----------:|--------:|-------:|
| Logistic Regression | 0.614 | 0.633 | 0.595 | 0.847 | 0.655 |
| Random Forest | 0.759 | 0.847 | 0.688 | 0.926 | 0.831 |
| LSTM | 0.830 | 0.886 | 0.781 | 0.966 | 0.932 |
| Transformer | 0.863 | 0.815 | 0.916 | 0.975 | 0.950 |
| **MSTAN (Ours)** | **0.848** | **0.851** | **0.845** | **0.968** | **0.936** |

> **Key insight:** MSTAN achieves the most balanced precision–recall trade-off (0.845 / 0.851) among all models. The Transformer has higher F1 but sacrifices recall; the LSTM has higher recall but lower precision. MSTAN is the only model where both are nearly equal — critical for real-world churn intervention where both false positives and false negatives carry cost.

### Visualizations

All generated automatically by `run_all.py`:

| Figure | Description |
|--------|-------------|
| `comparison_bar.png` | Grouped bar chart — all models across F1, Recall, Precision, AUC, PR-AUC |
| `radar_chart.png` | Spider chart — model strengths at a glance |
| `multi_roc.png` | Overlaid ROC curves for all 5 models |
| `multi_pr.png` | Overlaid Precision-Recall curves |
| `training_history.png` | 2×2 grid — train/val loss, F1, and AUC across epochs |
| `*_confusion.png` | Per-model confusion matrices |

## Features

- **Temporal sequence generation** from static Telco records with churn-correlated behaviour patterns
- **Preprocessing**: validation, missing-value handling, one-hot encoding, standard scaling
- **Models**: MSTAN (novel), Transformer encoder, LSTM, Logistic Regression, Random Forest
- **Training**: Focal Loss, AdamW, LR warmup + cosine annealing, early stopping, gradient clipping, GPU support
- **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, F1-optimal threshold selection
- **Interpretability**: attention weight extraction and heatmap visualization
- **Deployment** (optional): FastAPI `/predict-churn` endpoint

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the IBM Telco Customer Churn CSV at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (or update `configs/base_config.yaml`).

## Quickstart

### Run everything (recommended)

Trains all 5 models, generates reports and publication-quality figures:

```bash
python run_all.py
```

### Train a single model

```bash
python -m src.training.train --config configs/base_config.yaml --model mstan
python -m src.training.train --config configs/base_config.yaml --model transformer
python -m src.training.train --config configs/base_config.yaml --model lstm
```

### Evaluate a checkpoint

```bash
python -m src.evaluation.evaluate --config configs/base_config.yaml --checkpoint artifacts/checkpoints/best_mstan.pt --model mstan
```

### FastAPI (optional)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Example request to `/predict-churn`:

```json
{
  "sequence": [
    {
      "tenure": 12, "MonthlyCharges": 70, "TotalCharges": 840,
      "gender": "Female", "SeniorCitizen": "0", "Partner": "Yes",
      "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
      "InternetService": "Fiber optic", "OnlineSecurity": "No",
      "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No",
      "StreamingTV": "Yes", "StreamingMovies": "Yes",
      "Contract": "Month-to-month", "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check"
    }
  ]
}
```

Response: `{"churn_probability": 0.82, "risk": "high"}`

## Project Structure

```
Churn_AI/
├── configs/
│   └── base_config.yaml          # All hyperparameters and paths
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── src/
│   ├── data/
│   │   ├── preprocessing.py      # Cleaning, encoding, scaling
│   │   ├── sequence_generator.py # Churn-correlated temporal sequences
│   │   └── datamodule.py         # PyTorch datasets and dataloaders
│   ├── models/
│   │   ├── mstan.py              # Novel MSTAN architecture
│   │   ├── transformer.py        # Vanilla Transformer encoder
│   │   └── lstm.py               # LSTM baseline
│   ├── training/
│   │   ├── train.py              # Training loop
│   │   ├── focal_loss.py         # Focal Loss implementation
│   │   ├── metrics.py            # Metrics + threshold optimization
│   │   ├── early_stopping.py     # Patience-based early stopping
│   │   └── utils.py              # Seeds, device, checkpointing
│   ├── evaluation/
│   │   ├── evaluate.py           # Model evaluation pipeline
│   │   ├── visualize.py          # Publication-quality plots
│   │   └── interpretability.py   # Attention extraction
│   └── api/
│       └── main.py               # FastAPI inference endpoint
├── run_all.py                    # One-click orchestrator
├── artifacts/
│   ├── checkpoints/              # Saved model weights
│   ├── reports/                  # JSON metrics
│   └── figures/                  # Generated plots
└── requirements.txt
```

## Configuration

All settings are in `configs/base_config.yaml`:

| Section | Key Parameters |
|---------|---------------|
| **data** | `seq_len: 12`, `drift_std: 0.03`, `val_size: 0.15`, `test_size: 0.15` |
| **model** | `d_model: 64`, `n_heads: 4`, `num_layers: 2`, `scales: [1,2,4]` |
| **training** | `lr: 0.0008`, `loss: focal`, `focal_alpha: 0.75`, `warmup_epochs: 5`, `scheduler: cosine` |

## Technical Notes

- **Reproducibility**: Seeds set across `torch`, `numpy`, and `random`
- **GPU**: Automatically used if available; override via `device: cpu` in config
- **Threshold optimization**: F1-optimal threshold found on the validation set, applied to test
- **Class imbalance**: Handled by Focal Loss (not class weights) — more robust for neural networks
- **Sequence generation**: Churners exhibit rising charges, stagnating tenure, and mid-sequence service downgrades; non-churners show stable trajectories
