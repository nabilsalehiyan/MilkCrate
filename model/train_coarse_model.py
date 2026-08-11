"""
train_coarse_model.py
======================
Trains the Stage-1 "gate" classifier that separates Electronic/Dance music
from other broad genres (Hip-Hop/R&B, Rock, Pop/Vocal, Jazz/Acoustic, etc.)

Unlike train_model.py (which uses the precomputed BeatsDataset CSV), this
trains from RAW AUDIO FILES, because we need to extract features ourselves
(no precomputed-feature dataset for non-electronic genres was bundled).

-------------------------------------------------------------------------
HOW TO BUILD YOUR TRAINING SET (no big dataset download needed)
-------------------------------------------------------------------------
Create a folder structure like this using tracks you already have:

    coarse_training_data/
        Electronic_Dance/   <- house, techno, trance, dubstep, DnB, etc.
            track1.mp3
            track2.flac
            ...
        HipHop_RnB/
            frank_ocean_track.mp3
            ...
        Rock_Alternative/
            ...
        Pop_Vocal/
            ...
        Jazz_Acoustic_Classical/
            ...
        (add more categories as needed — anything that should NOT be
         routed into the EDM subgenre model)

Guidelines:
  - Aim for at least ~20-30 tracks per category for a usable baseline;
    more is better. Doesn't need to be huge — this is a coarse gate,
    not a fine-grained model.
  - The "Electronic_Dance" category can reuse a sample of your own
    house/techno/etc. tracks (or any electronic tracks you have).
  - Categories are entirely up to you — just make sure "Electronic_Dance"
    is spelled exactly that way (it's the label main.py checks for).

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
    python train_coarse_model.py --data-dir coarse_training_data
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
from feature_extract import extract_features_from_file, DEFAULT_SR, DEFAULT_MAX_SECS  # noqa: E402

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


AUDIO_EXTENSIONS = {".mp3", ".flac", ".aiff", ".aif", ".wav", ".m4a", ".aac", ".ogg", ".alac"}


def get_model(kind: str, random_state: int = 42):
    kind = (kind or "lgbm").lower()
    if kind == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
            )
        except Exception:
            print("[INFO] LightGBM not available; falling back to RandomForest.")
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=random_state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Folder with one subfolder per coarse category, containing audio files")
    ap.add_argument("--model-kind", default="lgbm", choices=["lgbm", "rf"])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--sr", type=int, default=DEFAULT_SR)
    ap.add_argument("--max-secs", type=int, default=DEFAULT_MAX_SECS)
    ap.add_argument("--out-dir", default="artifacts")
    args = ap.parse_args()

    categories = sorted([
        d for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ])
    if "Electronic_Dance" not in categories:
        print("[WARNING] No 'Electronic_Dance' folder found. The gate needs this "
              "exact label to route tracks to the fine-grained EDM model.")

    print(f"[INFO] Found categories: {categories}")

    rows = []
    labels = []
    for cat in categories:
        cat_dir = os.path.join(args.data_dir, cat)
        files = [f for f in os.listdir(cat_dir)
                 if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS]
        print(f"[INFO] {cat}: {len(files)} files")
        for fn in files:
            path = os.path.join(cat_dir, fn)
            try:
                feats = extract_features_from_file(path, sr=args.sr, max_secs=args.max_secs)
                feats.pop("_duration_analyzed_sec", None)
                rows.append(feats)
                labels.append(cat)
            except Exception as e:
                print(f"[WARN] Failed {path}: {e}")

    if not rows:
        raise SystemExit("[ERROR] No training data extracted. Check --data-dir structure.")

    feat_df = pd.DataFrame(rows)
    feature_cols = sorted(feat_df.columns.tolist())
    feat_df = feat_df[feature_cols]

    X = feat_df.to_numpy(dtype=np.float32)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"\n[INFO] Total samples: {len(y)}, classes: {list(le.classes_)}")

    if len(y) < 10 or min(np.bincount(y)) < 2:
        print("[WARNING] Very small dataset — skipping train/test split, "
              "training on all data (no held-out eval).")
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
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
    print(f"\n=== Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1m:.4f}")

    target_names = [str(c) for c in le.classes_]
    print("\n=== Per-class report ===")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.out_dir, "coarse_model.joblib"))
    joblib.dump(le, os.path.join(args.out_dir, "coarse_label_encoder.joblib"))
    with open(os.path.join(args.out_dir, "coarse_feature_columns.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    cm_df.to_csv(os.path.join(args.out_dir, "coarse_confusion_matrix.csv"))

    print(f"\n[INFO] Saved coarse model    -> {args.out_dir}/coarse_model.joblib")
    print(f"[INFO] Saved coarse encoder  -> {args.out_dir}/coarse_label_encoder.joblib")
    print(f"[INFO] Saved feature columns -> {args.out_dir}/coarse_feature_columns.json")


if __name__ == "__main__":
    main()
