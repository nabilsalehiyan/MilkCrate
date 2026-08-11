"""
train_from_custom_dataset.py
==============================
Trains the fine-grained EDM subgenre model from YOUR OWN labeled audio
files, optionally merged with the base BeatsDataset CSV.

-------------------------------------------------------------------------
DATASET FORMAT
-------------------------------------------------------------------------
Organize your manually-tagged library like this:

    my_dataset/
        DeepHouse/
            track1.flac
            track2.mp3
            ...
        TechHouse/
            track1.flac
            ...
        ProgressiveHouse/
            ...
        (one folder per subgenre — folder name = label)

Folder names should match the 22-class taxonomy used elsewhere in this
app (DeepHouse, TechHouse, House, ProgressiveHouse, Minimal, BigRoom,
ElectroHouse, FutureHouse, IndieDanceNuDisco, HardDance, Techno,
HardcoreHardTechno, Trance, PsyTrance, Dance, ElectronicaDowntempo,
FunkRAndB, Breaks, DrumAndBass, Dubstep, GlitchHop, HipHop) — but you can
use your own labels too; just update FAMILY_MAP below (or in
app/main.py / model/train_model.py) so the family-level grouping still
works for any new labels.

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
# Train from your dataset only:
    python train_from_custom_dataset.py --data-dir my_dataset

# Merge with the base BeatsDataset CSV (recommended if your dataset is
# small — gives the model more examples per class overall):
    python train_from_custom_dataset.py --data-dir my_dataset \\
        --base-csv beatsdataset_full.csv --exclude-classes ReggaeDub

Feature extraction uses the exact same pipeline as live scanning
(app/feature_extract.py), including the BPM-tag and bpmessentia fixes —
so your custom dataset will be schema-consistent with live predictions.
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
from feature_extract import extract_features_from_file, normalize_csv_columns, DEFAULT_SR, DEFAULT_MAX_SECS  # noqa: E402

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


AUDIO_EXTENSIONS = {".mp3", ".flac", ".aiff", ".aif", ".wav", ".m4a", ".aac", ".ogg", ".alac"}
NON_FEATURE_COLS = {"unnamed: 0", "class", "id", "family"}

# Default taxonomy -> family mapping; extend this if you use custom labels
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


def extract_custom_dataset(data_dir: str, sr: int, max_secs: int, cache_path: str = None):
    """
    Walk data_dir/<label>/*.audio, extract features for each file.
    Caches results to a parquet/json so re-runs don't re-extract everything.
    Returns a DataFrame with feature columns + 'class'.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Loading cached extraction from {cache_path}")
        return pd.read_parquet(cache_path)

    labels = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    print(f"[INFO] Found {len(labels)} label folders: {labels}")

    rows = []
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        files = [f for f in os.listdir(label_dir)
                 if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS]
        print(f"[INFO] {label}: {len(files)} files")
        for fn in files:
            path = os.path.join(label_dir, fn)
            try:
                feats = extract_features_from_file(path, sr=sr, max_secs=max_secs)
                feats.pop("_duration_analyzed_sec", None)
                feats.pop("_bpm_source", None)
                feats["class"] = label
                rows.append(feats)
            except Exception as e:
                print(f"[WARN] Failed {path}: {e}")

    if not rows:
        raise SystemExit("[ERROR] No audio files found/extracted. Check --data-dir structure.")

    df = pd.DataFrame(rows)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"[INFO] Cached extraction -> {cache_path}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Folder with one subfolder per subgenre label, containing audio files")
    ap.add_argument("--base-csv", default=None,
                    help="Optional: base BeatsDataset CSV to merge with your custom data")
    ap.add_argument("--exclude-classes", nargs="*", default=["ReggaeDub"],
                    help="Classes to drop from the base CSV (default: ReggaeDub)")
    ap.add_argument("--model-kind", default="lgbm", choices=["lgbm", "rf"])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--top-k-eval", type=int, default=3)
    ap.add_argument("--sr", type=int, default=DEFAULT_SR)
    ap.add_argument("--max-secs", type=int, default=DEFAULT_MAX_SECS)
    ap.add_argument("--cache", default="artifacts/custom_dataset_features.parquet",
                    help="Where to cache extracted features (delete to force re-extraction)")
    ap.add_argument("--out-dir", default="artifacts")
    args = ap.parse_args()

    custom_df = extract_custom_dataset(args.data_dir, args.sr, args.max_secs, args.cache)
    print(f"[INFO] Custom dataset: {len(custom_df)} rows, "
          f"classes={sorted(custom_df['class'].unique())}")

    if args.base_csv:
        base_df = pd.read_csv(args.base_csv)
        base_df.columns = normalize_csv_columns(list(base_df.columns))
        if args.exclude_classes:
            base_df = base_df[~base_df["class"].isin(args.exclude_classes)]

        # Align feature columns: union of both, fill missing with NaN
        feature_cols = sorted(set(
            [c for c in custom_df.columns if c not in NON_FEATURE_COLS] +
            [c for c in base_df.columns if c not in NON_FEATURE_COLS]
        ))
        for c in feature_cols:
            if c not in custom_df.columns:
                custom_df[c] = np.nan
            if c not in base_df.columns:
                base_df[c] = np.nan

        combined = pd.concat(
            [custom_df[feature_cols + ["class"]], base_df[feature_cols + ["class"]]],
            ignore_index=True,
        )
        print(f"[INFO] Merged with base CSV: {len(combined)} total rows, "
              f"{len(feature_cols)} features")
    else:
        feature_cols = sorted([c for c in custom_df.columns if c not in NON_FEATURE_COLS])
        combined = custom_df[feature_cols + ["class"]]

    X = combined[feature_cols].to_numpy(dtype=np.float32)
    y_raw = combined["class"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    counts = pd.Series(y_raw).value_counts()
    min_count = counts.min()
    if min_count < 2:
        print("[WARNING] Some classes have <2 examples — training on all "
              "data, no held-out eval.")
        X_train, y_train = X, y
        X_test, y_test = X, y
    else:
        stratify = y if min_count >= 2 and (counts >= 2).all() else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=stratify
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

    # Build family map: use defaults for known labels, "Unknown" for new ones
    family_map = {label: FAMILY_MAP.get(label, "Unknown") for label in le.classes_}

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
    print(f"[INFO] Saved family map  -> {args.out_dir}/family_map.json "
          f"(review this — any new/custom labels got family='Unknown')")
    print("[INFO] Restart the app to use the updated model.")


if __name__ == "__main__":
    main()
