"""FastAPI spam classifier API."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from email_text import extract_text_from_bytes, normalize_text
from heuristic_spam import heuristic_spam_mass

MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parent / "artifacts" / "model.joblib"))


def _spam_decision_threshold() -> float:
    """Probability above this counts as spam. Default 0.65 reduces false positives on borderline mail."""
    try:
        t = float(os.environ.get("SPAM_THRESHOLD", "0.65"))
    except ValueError:
        t = 0.65
    return max(0.01, min(0.99, t))


app = FastAPI(title="Spam Detector API", version="1.0.0")

_cors = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle: dict | None = None


def _load_bundle() -> dict:
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.is_file():
            raise RuntimeError(
                f"Model not found at {MODEL_PATH}. Run: python scripts/download_data.py && python train.py"
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


@app.on_event("startup")
def startup() -> None:
    _load_bundle()


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ClassifyResponse(BaseModel):
    label: str
    spam_probability: float
    version: str
    spam_threshold: float
    ml_spam_probability: float


def _response_for_text(text: str) -> ClassifyResponse:
    bundle = _load_bundle()
    pipeline = bundle["pipeline"]
    version = bundle.get("version", "unknown")
    if not text:
        raise HTTPException(status_code=400, detail="Empty text after normalization")
    proba = pipeline.predict_proba([text])[0]
    classes = list(pipeline.classes_)
    if len(classes) != 2:
        raise HTTPException(status_code=500, detail="Expected binary classifier")
    spam_idx = classes.index("spam")
    ml_spam_p = float(proba[spam_idx])
    extra = heuristic_spam_mass(text)
    spam_p = min(1.0, ml_spam_p + extra * (1.0 - ml_spam_p))
    thr = _spam_decision_threshold()
    label = "spam" if spam_p >= thr else "ham"
    return ClassifyResponse(
        label=label,
        spam_probability=spam_p,
        version=version,
        spam_threshold=thr,
        ml_spam_probability=ml_spam_p,
    )


@app.post("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/classify", response_model=ClassifyResponse)
def classify(body: ClassifyRequest) -> ClassifyResponse:
    text = normalize_text(body.text)
    return _response_for_text(text)


@app.post("/classify/eml", response_model=ClassifyResponse)
async def classify_eml(file: UploadFile = File(...)) -> ClassifyResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        text = extract_text_from_bytes(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse message: {exc}") from exc
    return _response_for_text(text)
