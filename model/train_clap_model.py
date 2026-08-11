"""
Train LightGBM classifier on top of CLAP embeddings.
Run extract_clap_features.py first, or pass --data-dir to do both.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

FAMILY_DEFAULTS = {
    "Tech_House": "House", "Minimal_deep": "House", "Afro_house": "House",
    "Classic_house": "House", "Melodic_house": "House", "DeepHouse": "House",
    "Progressive_House": "House", "House": "House", "ElectroHouse": "House",
    "FutureHouse": "House", "BigRoom": "House", "Bass_house": "Bass",
    "Techno": "Techno", "HardDance": "Techno",
    "Drum_and_bass": "Bass", "UKG": "Bass", "Dubstep": "Bass", "Breaks": "Bass",
    "Trance": "Trance", "PsyTrance": "Trance",
    "Dance": "Electronic", "Downtempo": "Electronic",
    "HipHop_RnB": "Other", "Rock_Alternative": "Other",
    "Pop_Vocal": "Other", "Jazz_Acoustic_Classical": "Other", "Other": "Other",
}

def train(features_path, out_dir):
    print(f"[INFO] Loading features from {features_path}")
    df = pd.read_parquet(features_path)

    feature_cols = [c for c in df.columns if c.startswith("clap_")]
    X = df[feature_cols].values
    y = df['label'].values

    print(f"[INFO] {len(df)} tracks, {len(feature_cols)} features, {len(set(y))} classes")
    print(pd.Series(y).value_counts().to_string())

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=3,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== Evaluation ===")
    print(f"Accuracy: {acc:.4f}")

    # Top-3
    proba = model.predict_proba(X_test)
    top3 = np.argsort(proba, axis=1)[:, -3:]
    top3_acc = np.mean([y_test[i] in top3[i] for i in range(len(y_test))])
    print(f"Top-3 Acc: {top3_acc:.4f}")

    print(f"\n=== Per-class report ===")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_,
                                digits=4, zero_division=0))

    # Save
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "genre_model.joblib"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.joblib"))

    feature_cols_path = os.path.join(out_dir, "feature_columns.json")
    with open(feature_cols_path, 'w') as f:
        json.dump(feature_cols, f)

    family_map = {label: FAMILY_DEFAULTS.get(label, "Unknown") for label in le.classes_}
    with open(os.path.join(out_dir, "family_map.json"), 'w') as f:
        json.dump(family_map, f, indent=2)

    # Save a flag so the app knows to use CLAP features
    with open(os.path.join(out_dir, "model_type.json"), 'w') as f:
        json.dump({"type": "clap", "model_id": "laion/larger_clap_music"}, f)

    print(f"\n[INFO] Saved model -> {out_dir}/genre_model.joblib")
    print(f"[INFO] Model type: CLAP embeddings")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="artifacts/clap_features.parquet")
    ap.add_argument("--data-dir", default=None, help="If set, extract features first")
    ap.add_argument("--out-dir", default="artifacts")
    ap.add_argument("--max-secs", type=int, default=60)
    args = ap.parse_args()

    if args.data_dir:
        import torch
        from extract_clap_features import extract_dataset
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        df = extract_dataset(args.data_dir, args.features, device, args.max_secs)

    train(args.features, args.out_dir)
