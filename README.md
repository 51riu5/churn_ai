# Deep Learning–Driven Customer Churn Forecasting

Transformer-based temporal modeling for the IBM Telco Customer Churn dataset with end-to-end preprocessing, baselines, training, evaluation, interpretability, and optional FastAPI deployment.

## Features
- Temporal sequence generation from static Telco records (configurable length)
- Preprocessing: validation, missing-value handling, categorical encoding, numeric scaling
- Models: Transformer encoder (main), LSTM, Logistic Regression, Random Forest
- Training: BCE loss, Adam/AdamW, cosine scheduler, early stopping, checkpointing, GPU support
- Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC plots
- Interpretability: attention weight extraction and visualization
- Deployment (optional): FastAPI `/predict-churn` endpoint

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the IBM Telco Customer Churn CSV from Kaggle and place it at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` (or update the path in `configs/base_config.yaml`).

## Quickstart (Transformer)
```bash
python -m src.training.train --config configs/base_config.yaml --model transformer
```

Evaluate a saved checkpoint:
```bash
python -m src.evaluation.evaluate --config configs/base_config.yaml --checkpoint artifacts/checkpoints/best_transformer.pt
```

Train baselines:
```bash
python -m src.baselines.train_sklearn --config configs/base_config.yaml
```

Run FastAPI (optional):
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Project Structure
- `configs/` – YAML configs with data/model/training defaults
- `src/data/` – loading, preprocessing, temporal sequence generation, datasets
- `src/models/` – Transformer, LSTM, and sklearn baselines
- `src/training/` – train loop, metrics, early stopping, utilities
- `src/evaluation/` – evaluation scripts, visualization, interpretability
- `src/api/` – FastAPI inference endpoint
- `artifacts/` – checkpoints, reports, and figures (created on demand)

## Notes
- Reproducibility: seeds set across `torch`, `numpy`, and `random`
- GPU automatically used if available; override via `--device cpu`
- Design choices are commented for viva/academic defense
