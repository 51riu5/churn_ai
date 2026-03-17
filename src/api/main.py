"""
Churn AI — Interactive Dashboard Backend
=========================================
Serves the presentation dashboard + API endpoints for live prediction,
model comparison data, training histories, and figure images.

Usage (local):
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Usage (deployed):
    Render / Railway auto-detect the Procfile.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

# Optional: PyTorch is only needed for MSTAN inference (not for the dashboard)
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = PROJECT_ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
REPORTS = ARTIFACTS / "reports"
CHECKPOINTS = ARTIFACTS / "checkpoints"
STATIC = Path(__file__).resolve().parent / "static"
CONFIG_PATH = PROJECT_ROOT / "configs" / "base_config.yaml"

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(title="Churn AI Dashboard", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state (loaded at startup) ──────────────────────────────
_model = None
_preprocessor = None
_logreg = None
_device = None
_cfg: Dict[str, Any] = {}
_seq_len: int = 12


# ── Startup ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global _model, _preprocessor, _logreg, _device, _cfg, _seq_len

    from src.config import load_config
    from src.data.preprocessing import (
        CATEGORICAL_COLS,
        NUMERIC_COLS,
        build_preprocessor,
        clean_telco_data,
        fit_preprocessor,
        validate_telco_columns,
    )

    _cfg = load_config(str(CONFIG_PATH))
    _seq_len = _cfg["data"].get("seq_len", 12)

    # 1. Load preprocessor (try pickle first, fallback to fit from CSV)
    pkl_path = CHECKPOINTS / "preprocessor.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            _preprocessor = pickle.load(f)
        print("[API] Loaded preprocessor from pickle")
    else:
        csv_path = PROJECT_ROOT / _cfg["data"]["path"]
        df = pd.read_csv(str(csv_path))
        validate_telco_columns(df, "Churn")
        df = clean_telco_data(df, "Churn")
        _preprocessor = build_preprocessor(CATEGORICAL_COLS, NUMERIC_COLS)
        _preprocessor = fit_preprocessor(_preprocessor, df, "Churn", "customerID")
        print("[API] Built preprocessor from CSV (no pickle found)")

    # 2. (Optional) Load MSTAN model — only if PyTorch is installed
    if HAS_TORCH:
        try:
            from src.models.mstan import MSTANChurnClassifier
            from src.training.utils import get_device

            _device = get_device(_cfg["training"].get("device", "auto"))
            ckpt = CHECKPOINTS / "best_mstan.pt"
            if ckpt.exists() and _preprocessor is not None:
                dummy = pd.DataFrame(
                    {col: ["Unknown"] for col in CATEGORICAL_COLS}
                    | {col: [0.0] for col in NUMERIC_COLS}
                )
                feature_dim = _preprocessor.transform(dummy).shape[1]
                mc = _cfg["model"]
                _model = MSTANChurnClassifier(
                    input_dim=feature_dim,
                    d_model=mc.get("d_model", 64),
                    nhead=mc.get("n_heads", 4),
                    num_layers=mc.get("num_layers", 2),
                    dim_feedforward=mc.get("dim_feedforward", 128),
                    dropout=mc.get("dropout", 0.1),
                    input_dropout=mc.get("input_dropout", 0.1),
                    scales=tuple(mc.get("scales", [1, 2, 4])),
                    conv_kernel_size=mc.get("conv_kernel_size", 3),
                ).to(_device)
                state = torch.load(str(ckpt), map_location=_device, weights_only=True)
                _model.load_state_dict(state)
                _model.eval()
                print(f"[API] Loaded MSTAN model ({feature_dim} features)")
        except Exception as e:
            print(f"[API] MSTAN loading skipped: {e}")
    else:
        print("[API] PyTorch not installed — MSTAN loading skipped (dashboard still works)")

    # 3. Train LogReg for live predictions (only needs sklearn)
    from sklearn.linear_model import LogisticRegression

    csv_path = PROJECT_ROOT / _cfg["data"]["path"]
    df = pd.read_csv(str(csv_path))
    validate_telco_columns(df, "Churn")
    df = clean_telco_data(df, "Churn")
    feature_df = df.drop(columns=["Churn", "customerID"])
    X = _preprocessor.transform(feature_df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    y = df["Churn"].values
    _logreg = LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear")
    _logreg.fit(X, y)
    print("[API] Trained LogReg for live predictions")

    port = os.environ.get("PORT", "8000")
    print(f"[API] Ready!")


# ── Schemas ────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    tenure: float = 12
    MonthlyCharges: float = 70
    TotalCharges: float = 840
    gender: str = "Female"
    SeniorCitizen: str = "0"
    Partner: str = "Yes"
    Dependents: str = "No"
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "Yes"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "Yes"
    StreamingMovies: str = "Yes"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"


# ── Routes ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_path = STATIC / "index.html"
    if not html_path.exists():
        raise HTTPException(404, "Dashboard HTML not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/comparison")
async def get_comparison():
    path = REPORTS / "all_models_comparison.json"
    if not path.exists():
        raise HTTPException(404, "No comparison data yet. Run run_all.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/histories")
async def get_histories():
    path = REPORTS / "training_histories.json"
    if not path.exists():
        raise HTTPException(404, "No training histories. Run run_all.py first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/figures/{filename}")
async def get_figure(filename: str):
    safe = Path(filename).name
    path = FIGURES / safe
    if not path.exists():
        raise HTTPException(404, f"Figure '{filename}' not found")
    return FileResponse(str(path), media_type="image/png")


@app.post("/api/predict")
async def predict(customer: CustomerData):
    if _logreg is None or _preprocessor is None:
        raise HTTPException(503, "Model not loaded. Run run_all.py first.")

    try:
        row = customer.model_dump()
        df = pd.DataFrame([row])
        encoded = _preprocessor.transform(df)
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        prob = float(_logreg.predict_proba(encoded)[:, 1][0])
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}")

    risk = "High" if prob >= 0.6 else "Medium" if prob >= 0.35 else "Low"
    color = "#ef4444" if risk == "High" else "#f59e0b" if risk == "Medium" else "#10b981"

    return {
        "probability": round(prob, 4),
        "percentage": round(prob * 100, 1),
        "risk": risk,
        "color": color,
    }
