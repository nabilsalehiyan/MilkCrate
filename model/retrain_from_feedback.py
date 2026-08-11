"""
retrain_from_feedback.py
=========================
Retrains the fine-grained EDM subgenre model using the original
BeatsDataset PLUS any corrections you've made via the app's Override
dropdown (logged to ~/.milkcrate/feedback.jsonl).

How it works:
  - Each correction record already contains the full ~92-feature vector
    extracted from YOUR track at scan time, plus the label you corrected
    it to.
  - These get appended as extra training rows alongside the base CSV.
  - Repeated corrections for the same track use only the most recent one.

This means every correction you make becomes real training data — the
model gradually adapts to your library's sound and your taxonomy
preferences over time.

Usage:
    python retrain_from_feedback.py
    python retrain_from_feedback.py --csv beatsdataset_full.csv --model-kind lgbm

Only corrections whose corrected_label is one of the 22 EDM subgenre
classes are used (corrections to Unsorted_LowConfidence or non-EDM coarse
labels are skipped for the fine model — those would need the coarse
model's retrain script instead).
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

# Must match app/main.py's EDM_SUBGENRES
EDM_SUBGENRES = {
    "BigRoom", "Breaks", "Dance", "DeepHouse", "DrumAndBass", "Dubstep",
    "ElectroHouse", "ElectronicaDowntempo", "FunkRAndB", "FutureHouse",
    "GlitchHop", "HardcoreHardTechno", "HardDance", "HipHop", "House",
    "IndieDanceNuDisco", "Minimal", "ProgressiveHouse", "PsyTrance",
    "TechHouse", "Techno", "Trance",
}

# Static subgenre -> family taxonomy (mirrors model/train_model.py)
FAMILY_MAP = {
    "Breaks": "Bass", "DrumAndBass": "Bass", "Dubstep": "Bass",
    "GlitchHop": "Bass", "HipHop": "Bass",
    "Dance": "Electronic", "ElectronicaDowntempo": "Electronic", "FunkRAndB": "Electronic",
    "HardcoreHardTechno": "HardTechno", "HardDance": "HardTechno", "Techno": "HardTechno",
    "BigRoom": "House", "DeepHouse": "House", "ElectroHouse": "House",
    "FutureHouse": "House", "House": "House", "IndieDanceNuDisco": "House",
    "Minimal": "House", "ProgressiveHouse": "House", "TechHouse": "House",
    "PsyTrance": "Trance", "Trance": "Trance",
}


def feedback_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".milkcrate", "feedback.jsonl")


def load_feedback_rows(feature_cols):
    """
    Load corrections and convert to rows matching the training CSV schema.
    Returns (rows_df, n_used, n_skipped).
    """
    fp = feedback_path()
    if not os.path.exists(fp):
        return pd.DataFrame(columns=feature_cols + ["class"]), 0, 0

    # Deduplicate: keep only the most recent correction per path
    latest = {}
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            latest[rec["path"]] = rec

    rows = []
    n_skipped = 0
    for rec in latest.values():
        label = rec.get("corrected_label")
        if label not in EDM_SUBGENRES:
            n_skipped += 1
            continue

        feats = rec.get("features", {})
        row = {}
        for c in feature_cols:
            row[c] = feats.get(c, np.nan)
        row["class"] = label
        rows.append(row)

    df = pd.DataFrame(rows, columns=feature_cols + ["class"]) if rows else pd.DataFrame(columns=feature_cols + ["class"])
    return df, len(rows), n_skipped


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
    return RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=random_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="beatsdataset_full.csv",
                    help="Base training CSV (precomputed BeatsDataset features)")
    ap.add_argument("--exclude-classes", nargs="*", default=["ReggaeDub"],
                    help="Classes to drop from the base CSV (default: ReggaeDub, "
                         "see earlier analysis of why)")
    ap.add_argument("--model-kind", default="lgbm", choices=["lgbm", "rf"])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--top-k-eval", type=int, default=3)
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--feedback-weight", type=float, default=3.0,
                    help="Replicate each feedback row this many times (rounded) "
                         "so user corrections carry more weight than base-dataset "
                         "rows during training. Default 3x.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    raw_cols = list(df.columns)
    norm_cols = normalize_csv_columns(raw_cols)
    df.columns = norm_cols

    if args.exclude_classes:
        df = df[~df["class"].isin(args.exclude_classes)]

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"[INFO] Base dataset: {len(df)} rows, {len(feature_cols)} features, "
          f"classes={sorted(df['class'].unique())}")

    feedback_df, n_used, n_skipped = load_feedback_rows(feature_cols)
    print(f"[INFO] Feedback: {n_used} correction(s) usable for fine model, "
          f"{n_skipped} skipped (non-EDM or Unsorted labels)")

    if n_used > 0:
        # Upweight feedback rows by repetition (simple, no extra deps)
        reps = max(1, round(args.feedback_weight))
        feedback_df = pd.concat([feedback_df] * reps, ignore_index=True)
        print(f"[INFO] Feedback rows repeated {reps}x for weighting "
              f"-> {len(feedback_df)} effective rows")

        combined = pd.concat(
            [df[feature_cols + ["class"]], feedback_df[feature_cols + ["class"]]],
            ignore_index=True,
        )
    else:
        combined = df[feature_cols + ["class"]]

    X = combined[feature_cols].to_numpy(dtype=np.float32)
    y_raw = combined["class"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    model = get_model(args.model_kind, args.random_state)
    X_train_df = pd.DataFrame(X_train, columns=feature_cols)
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    model.fit(X_train_df, y_train)

    y_pred = model.predict(X_test_df)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    print(f"\n=== Evaluation (holdout, includes feedback rows) ===")
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
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.out_dir, "genre_model.joblib"))
    joblib.dump(le, os.path.join(args.out_dir, "label_encoder.joblib"))
    with open(os.path.join(args.out_dir, "family_map.json"), "w") as f:
        json.dump(FAMILY_MAP, f, indent=2)
    with open(os.path.join(args.out_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    cm_df.to_csv(os.path.join(args.out_dir, "confusion_matrix.csv"))

    print(f"\n[INFO] Saved model       -> {args.out_dir}/genre_model.joblib")
    print(f"[INFO] Saved encoder     -> {args.out_dir}/label_encoder.joblib")
    print(f"[INFO] {n_used} feedback correction(s) incorporated "
          f"(at {args.feedback_weight}x weight)")
    print("[INFO] Restart the app to use the updated model.")


if __name__ == "__main__":
    main()
