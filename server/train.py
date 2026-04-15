#!/usr/bin/env python3
"""Train TF-IDF + calibrated logistic regression on SpamAssassin public corpus."""

from __future__ import annotations

import email
import os
from pathlib import Path

import joblib
from email.policy import default as email_policy_default
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from email_text import extract_text_from_message, normalize_text

DATA_RAW = Path(__file__).resolve().parent / "data" / "raw"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
VERSION = "1.0.0"


def label_for_top_folder(folder_name: str) -> str | None:
    n = folder_name.lower()
    if "spam" in n and "ham" not in n:
        return "spam"
    if "ham" in n:
        return "ham"
    return None


def load_samples(root: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.name.endswith(".tar.bz2"):
            continue
        rel = path.relative_to(root)
        if not rel.parts:
            continue
        top = rel.parts[0]
        lab = label_for_top_folder(top)
        if lab is None:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if not data.strip():
            continue
        try:
            msg = email.message_from_bytes(data, policy=email_policy_default)
            text = extract_text_from_message(msg)
        except Exception:
            text = normalize_text(data.decode("utf-8", errors="replace"))
        if not text:
            continue
        texts.append(text)
        labels.append(lab)
    return texts, labels


def build_pipeline() -> Pipeline:
    word_vec = TfidfVectorizer(
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
    )
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])
    njobs = int(os.environ.get("SKLEARN_N_JOBS", "1"))
    base_lr = LogisticRegression(
        class_weight="balanced",
        solver="saga",
        max_iter=5000,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3, n_jobs=njobs)
    return Pipeline([("features", features), ("clf", calibrated)])


def main() -> None:
    if not DATA_RAW.is_dir():
        raise SystemExit(f"Missing {DATA_RAW}. Run: python scripts/download_data.py")

    X, y = load_samples(DATA_RAW)
    if len(X) < 100:
        raise SystemExit(f"Too few samples ({len(X)}). Check corpus under {DATA_RAW}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    print(f"Training on {len(X_train)} messages ...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out = Path(os.environ.get("MODEL_PATH", ARTIFACTS / "model.joblib"))
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "version": VERSION}, out)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
