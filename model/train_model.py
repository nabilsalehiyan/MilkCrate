"""
train_model.py
===============
Trains the genre/subgenre classifier from beatsdataset_full.csv (which
already contains precomputed audio features per track — no audio files
needed for training).

Outputs:
  artifacts/genre_model.joblib       - trained classifier
  artifacts/label_encoder.joblib     - LabelEncoder for the 'class' (subgenre) labels
  artifacts/family_map.json          - subgenre -> family lookup (for hierarchical sort)
  artifacts/feature_columns.json     - canonical feature column order the model expects

Usage:
    python train_model.py --csv beatsdataset_full.csv --model-kind lgbm
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from feature_extract import normalize_csv_columns  # noqa: E402

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


NON_FEATURE_COLS = {"unnamed: 0", "class", "id", "family"}


def get_model(kind: str, random_state: int = 42):
    kind = (kind or "lgbm").lower()
    if kind == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
            )
        except Exception:
            print("[INFO] LightGBM not available; falling back to RandomForest.")
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=500,
        n_jobs=-1,
        random_state=random_state,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="beatsdataset_full.csv")
    ap.add_argument("--model-kind", default="lgbm", choices=["lgbm", "rf"])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--top-k-eval", type=int, default=3)
    ap.add_argument("--out-dir", default="artifacts")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns from {args.csv}")

    # Normalize column names to canonical lowercase (strip "N-" prefixes)
    raw_cols = list(df.columns)
    norm_cols = normalize_csv_columns(raw_cols)
    df.columns = norm_cols

    # Identify feature vs. non-feature columns
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"[INFO] Using {len(feature_cols)} feature columns")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["class"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    model = get_model(args.model_kind, args.random_state)
    # Use DataFrame so the model records feature_names_in_
    X_train_df = pd.DataFrame(X_train, columns=feature_cols)
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    model.fit(X_train_df, y_train)

    y_pred = model.predict(X_test_df)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"\n=== Evaluation (holdout) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1m:.4f}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_df)
        k = min(args.top_k_eval, proba.shape[1])
        hits = sum(
            1 for i, p in enumerate(proba)
            if y_test[i] in np.argsort(p)[::-1][:k]
        )
        print(f"Top-{k} Acc: {hits / len(y_test):.4f}")

    target_names = [str(c) for c in le.classes_]
    print("\n=== Per-class report ===")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

    # Subgenre -> family map (static taxonomy from BeatsDataset families)
    family_map = {
        "Breaks": "Bass", "DrumAndBass": "Bass", "Dubstep": "Bass",
        "GlitchHop": "Bass", "HipHop": "Bass", "ReggaeDub": "Bass",
        "Dance": "Electronic", "ElectronicaDowntempo": "Electronic", "FunkRAndB": "Electronic",
        "HardcoreHardTechno": "HardTechno", "HardDance": "HardTechno", "Techno": "HardTechno",
        "BigRoom": "House", "DeepHouse": "House", "ElectroHouse": "House",
        "FutureHouse": "House", "House": "House", "IndieDanceNuDisco": "House",
        "Minimal": "House", "ProgressiveHouse": "House", "TechHouse": "House",
        "PsyTrance": "Trance", "Trance": "Trance",
    }

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.out_dir, "genre_model.joblib"))
    joblib.dump(le, os.path.join(args.out_dir, "label_encoder.joblib"))
    with open(os.path.join(args.out_dir, "family_map.json"), "w") as f:
        json.dump(family_map, f, indent=2)
    with open(os.path.join(args.out_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    cm_df.to_csv(os.path.join(args.out_dir, "confusion_matrix.csv"))

    print(f"\n[INFO] Saved model       -> {args.out_dir}/genre_model.joblib")
    print(f"[INFO] Saved encoder     -> {args.out_dir}/label_encoder.joblib")
    print(f"[INFO] Saved family map  -> {args.out_dir}/family_map.json")
    print(f"[INFO] Saved feature cols-> {args.out_dir}/feature_columns.json")


if __name__ == "__main__":
    main()
