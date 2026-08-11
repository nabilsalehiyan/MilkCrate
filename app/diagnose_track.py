"""
diagnose_track.py
==================
Diagnostic tool: extracts features from a real audio file and compares
them against the per-genre feature distributions in the training dataset,
to identify whether predictions are off due to a feature-extraction
mismatch vs. a model-quality issue.

Usage:
    python diagnose_track.py "/path/to/track.flac"
"""
import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from feature_extract import extract_features_from_file, normalize_csv_columns
from classifier import GenreClassifier

TRAIN_CSV = os.path.join(os.path.dirname(__file__), "..", "model", "beatsdataset_full.csv")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_track.py <audio_file>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Extracting features from: {path}\n")
    feats = extract_features_from_file(path)

    clf = GenreClassifier(os.path.join(os.path.dirname(__file__), "..", "model", "artifacts"))
    pred = clf.predict(feats, top_k=5)
    print("=== Prediction ===")
    print(f"Subgenre: {pred['subgenre']}  (family: {pred['family']})")
    print(f"BPM: {pred['bpm']}")
    print("Top-5:")
    for label, p in pred['top_k']:
        print(f"  {label}: {p:.1%}")

    # Compare key features to training distributions for predicted class
    # and for House (since user expects house-like tracks)
    train = pd.read_csv(TRAIN_CSV)
    train.columns = normalize_csv_columns(list(train.columns))

    compare_classes = list(dict.fromkeys([pred['subgenre'], "House", "TechHouse", "DeepHouse"]))

    key_feats = [
        "bpm", "danceability", "spectralcentroidm", "energym", "zcrm",
        "mfccs1m", "mfccs2m", "chromavector1m", "onset_rate",
        "beats_loudness.mean",
    ]

    print("\n=== Feature comparison (your track vs. training class means) ===")
    header = f"{'feature':<22}" + f"{'YOUR TRACK':>14}" + "".join(f"{c:>16}" for c in compare_classes)
    print(header)
    for feat in key_feats:
        row = f"{feat:<22}"
        val = feats.get(feat, float('nan'))
        row += f"{val:>14.3f}"
        for c in compare_classes:
            sub = train[train['class'] == c]
            if feat in sub.columns:
                m = sub[feat].mean()
                row += f"{m:>16.3f}"
            else:
                row += f"{'N/A':>16}"
        print(row)

    print("\n[INFO] If 'YOUR TRACK' values are wildly different in scale from ALL")
    print("training class columns (e.g. off by 10x, 100x, or sign-flipped),")
    print("that points to a feature-extraction/normalization mismatch.")
    print("If values are roughly in-range but just closer to the wrong class,")
    print("that points to a model accuracy issue instead.")


if __name__ == "__main__":
    main()
