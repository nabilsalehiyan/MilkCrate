"""
classifier.py
==============
Loads the trained model + label encoder + family map, and predicts
subgenre (and parent family) from an extracted feature dict.

Also provides TwoStageClassifier, which combines:
  Stage 1 (coarse_classifier.CoarseClassifier) - "is this Electronic/Dance?"
  Stage 2 (this GenreClassifier)               - EDM subgenre/family

If the coarse model isn't trained yet, TwoStageClassifier falls back to
running Stage 2 only (with a confidence threshold) — see main.py for the
"Unsorted/Low Confidence" bucket logic either way.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib


class GenreClassifier:
    def __init__(self, artifacts_dir: str):
        self.model = joblib.load(os.path.join(artifacts_dir, "genre_model.joblib"))
        self.label_encoder = joblib.load(os.path.join(artifacts_dir, "label_encoder.joblib"))
        with open(os.path.join(artifacts_dir, "feature_columns.json")) as f:
            self.feature_cols: List[str] = json.load(f)
        family_map_path = os.path.join(artifacts_dir, "family_map.json")
        self.family_map: Dict[str, str] = {}
        if os.path.exists(family_map_path):
            with open(family_map_path) as f:
                self.family_map = json.load(f)
        # Load scaler if present (used by enhanced model)
        scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
        self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    def _feature_row(self, features: Dict[str, float]) -> pd.DataFrame:
        row = {c: features.get(c, np.nan) for c in self.feature_cols}
        df = pd.DataFrame([row], columns=self.feature_cols)
        if self.scaler is not None:
            df = pd.DataFrame(self.scaler.transform(df), columns=self.feature_cols)
        return df

    def predict(self, features: Dict[str, float], top_k: int = 3) -> Dict:
        X = self._feature_row(features)

        pred_idx = int(self.model.predict(X)[0])
        subgenre = str(self.label_encoder.inverse_transform([pred_idx])[0])
        family = self.family_map.get(subgenre, "Unknown")

        top_labels: List[Tuple[str, float]] = []
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            k = min(top_k, len(proba))
            idxs = np.argsort(proba)[::-1][:k]
            for i in idxs:
                label = str(self.label_encoder.inverse_transform([int(i)])[0])
                top_labels.append((label, float(proba[i])))
        else:
            top_labels = [(subgenre, 1.0)]

        return {
            "subgenre": subgenre,
            "family": family,
            "top_k": top_labels,
            "bpm": features.get("bpm"),
        }


# ---------------------------------------------------------------------------
# Two-stage: coarse gate -> fine EDM model, with confidence-threshold fallback
# ---------------------------------------------------------------------------
UNSORTED_LABEL = "Unsorted_LowConfidence"


class TwoStageClassifier:
    """
    Combines the coarse Electronic/Dance gate with the fine-grained EDM
    subgenre model, plus a confidence threshold as a final safety net.

    Outcomes for a track's "subgenre"/"family":
      - A real EDM subgenre/family (passed both gates confidently)
      - The coarse model's non-electronic label (e.g. "HipHop_RnB"),
        used as both subgenre and family, if coarse model is available
        and confidently says "not electronic"
      - UNSORTED_LABEL, if confidence is too low at either stage
        (or the coarse model isn't trained yet and the fine model's
         top prediction is below the confidence threshold)
    """

    def __init__(
        self,
        artifacts_dir: str,
        coarse_confidence_threshold: float = 0.45,
        fine_confidence_threshold: float = 0.30,
    ):
        self.fine = GenreClassifier(artifacts_dir)
        self.coarse_confidence_threshold = coarse_confidence_threshold
        self.fine_confidence_threshold = fine_confidence_threshold

        self.coarse: Optional["CoarseClassifier"] = None
        try:
            from coarse_classifier import CoarseClassifier
            self.coarse = CoarseClassifier(artifacts_dir)
        except Exception:
            # Coarse model not trained yet — TwoStageClassifier will fall
            # back to fine-model-only with a confidence threshold.
            self.coarse = None

    @property
    def family_map(self) -> Dict[str, str]:
        return self.fine.family_map

    def predict(self, features: Dict[str, float], top_k: int = 3) -> Dict:
        # --- Stage 1: coarse gate ---
        if self.coarse is not None:
            coarse_result = self.coarse.predict(features)
            if not coarse_result["is_electronic"]:
                if coarse_result["confidence"] >= self.coarse_confidence_threshold:
                    label = coarse_result["coarse_label"]
                    return {
                        "subgenre": label,
                        "family": label,
                        "top_k": [(label, coarse_result["confidence"])],
                        "bpm": features.get("bpm"),
                        "stage": "coarse",
                        "coarse_confidence": coarse_result["confidence"],
                    }
                else:
                    # Coarse model unsure AND said non-electronic -> Unsorted
                    return {
                        "subgenre": UNSORTED_LABEL,
                        "family": UNSORTED_LABEL,
                        "top_k": [(coarse_result["coarse_label"], coarse_result["confidence"])],
                        "bpm": features.get("bpm"),
                        "stage": "coarse_lowconf",
                        "coarse_confidence": coarse_result["confidence"],
                    }
            # else: coarse says electronic -> fall through to fine model

        # --- Stage 2: fine EDM subgenre model ---
        fine_result = self.fine.predict(features, top_k=top_k)
        top1_conf = fine_result["top_k"][0][1] if fine_result["top_k"] else 0.0

        if top1_conf < self.fine_confidence_threshold:
            return {
                "subgenre": UNSORTED_LABEL,
                "family": UNSORTED_LABEL,
                "top_k": fine_result["top_k"],
                "bpm": fine_result["bpm"],
                "stage": "fine_lowconf",
                "fine_confidence": top1_conf,
            }

        fine_result["stage"] = "fine"
        fine_result["fine_confidence"] = top1_conf
        return fine_result
