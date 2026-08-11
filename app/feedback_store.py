"""
feedback_store.py
==================
Logs user corrections (override dropdown changes) along with the track's
already-extracted features. This builds a growing, real-world-labeled
dataset from YOUR library that can be used to retrain/fine-tune the model
over time (see model/retrain_from_feedback.py).

Stored as JSONL (one JSON object per line) for easy appending and easy
merging with the original training CSV.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional


def feedback_path() -> str:
    base = os.path.join(os.path.expanduser("~"), ".milkcrate")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "feedback.jsonl")


def log_correction(
    path: str,
    features: Dict[str, float],
    predicted_subgenre: Optional[str],
    predicted_family: Optional[str],
    corrected_label: str,
    confidence: Optional[float] = None,
):
    """
    Append one correction record. `corrected_label` is whatever the user
    set the Override dropdown to (an EDM subgenre, a non-EDM coarse
    category, or Unsorted_LowConfidence — only EDM subgenre corrections
    are useful for retraining the fine model, but we log everything for
    completeness / future coarse-model retraining too).
    """
    # Don't bother logging if the user didn't actually change anything
    if corrected_label == predicted_subgenre:
        return

    record = {
        "timestamp": time.time(),
        "path": path,
        "predicted_subgenre": predicted_subgenre,
        "predicted_family": predicted_family,
        "predicted_confidence": confidence,
        "corrected_label": corrected_label,
        "features": features,
    }

    with open(feedback_path(), "a") as f:
        f.write(json.dumps(record) + "\n")


def load_feedback() -> list:
    """Load all logged corrections."""
    fp = feedback_path()
    if not os.path.exists(fp):
        return []
    records = []
    with open(fp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def feedback_count() -> int:
    return len(load_feedback())
