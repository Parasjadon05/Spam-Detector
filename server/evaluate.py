#!/usr/bin/env python3
"""
Evaluate saved model on held-out data (same stratified 20% split as train.py).

Reports:
  - ML: sklearn predict() (default 0.5 decision)
  - ML+T: ML spam probability vs SPAM_THRESHOLD
  - Prod: combined score (API logic) vs SPAM_THRESHOLD
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from heuristic_spam import heuristic_spam_mass
from train import DATA_RAW, load_samples

MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parent / "artifacts" / "model.joblib"))


def spam_threshold() -> float:
    try:
        t = float(os.environ.get("SPAM_THRESHOLD", "0.65"))
    except ValueError:
        t = 0.65
    return max(0.01, min(0.99, t))


def combined_spam_p(ml_p: float, text: str) -> float:
    extra = heuristic_spam_mass(text)
    return min(1.0, ml_p + extra * (1.0 - ml_p))


def main() -> None:
    if not MODEL_PATH.is_file():
        raise SystemExit(f"No model at {MODEL_PATH}. Run: python train.py")
    if not DATA_RAW.is_dir():
        raise SystemExit(f"No data at {DATA_RAW}. Run: python scripts/download_data.py")

    X, y = load_samples(DATA_RAW)
    if len(X) < 100:
        raise SystemExit(f"Too few samples ({len(X)}).")

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    bundle = joblib.load(MODEL_PATH)
    pipe = bundle["pipeline"]
    classes = list(pipe.classes_)
    spam_i = classes.index("spam")

    y_true_spam = np.array([1 if lab == "spam" else 0 for lab in y_test])
    ml_proba = pipe.predict_proba(X_test)[:, spam_i]

    y_pred_sklearn = pipe.predict(X_test)

    thr = spam_threshold()
    y_pred_ml_thr = np.where(ml_proba >= thr, "spam", "ham")
    comb = np.array([combined_spam_p(float(p), t) for p, t in zip(ml_proba, X_test)])
    y_pred_prod = np.where(comb >= thr, "spam", "ham")

    n = len(y_test)
    spam_n = int(y_true_spam.sum())
    ham_n = n - spam_n

    print("=== Spam detector — held-out evaluation ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Test set: n={n} (spam={spam_n}, ham={ham_n}), same split as train.py (test_size=0.2, random_state=42)")
    print(f"SPAM_THRESHOLD (for ML+T and Prod): {thr}\n")

    def block(title: str, y_pred: np.ndarray) -> None:
        print(f"--- {title} ---")
        print(classification_report(y_test, y_pred, digits=4))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

    block("1) ML only — pipeline.predict() [sklearn default ~0.5]", y_pred_sklearn)

    try:
        auc = roc_auc_score(y_true_spam, ml_proba)
        print(f"ROC-AUC (spam positive class, ML P(spam)): {auc:.4f}\n")
    except ValueError as e:
        print(f"(ROC-AUC skipped: {e})\n")

    block(f"2) ML only — P(spam) >= {thr}", y_pred_ml_thr)
    block(f"3) Production — ML + heuristics, combined >= {thr}", y_pred_prod)


if __name__ == "__main__":
    main()
