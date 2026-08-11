"""
coarse_classifier.py
=====================
Stage 1 of the two-stage pipeline ("the gate", Cyanite-style).

Before running the detailed 23-class EDM subgenre model, this classifier
answers a coarser question: "Is this track Electronic/Dance music, and if
not, what broad category is it (Hip-Hop/R&B, Rock, Pop/Vocal, etc.)?"

This prevents non-electronic tracks (e.g. Frank Ocean) from being forced
into an EDM subgenre just because the fine-grained model has no "none of
these" option.

Artifacts expected in artifacts_dir:
    coarse_model.joblib
    coarse_label_encoder.joblib
    coarse_feature_columns.json

Uses the SAME feature_extract.extract_features() output as the fine model,
so no separate extraction pass is needed — one feature dict feeds both
stages.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib


# The label used by the fine-grained EDM model to mean "proceed to stage 2"
ELECTRONIC_LABEL = "Electronic_Dance"


class CoarseClassifier:
    def __init__(self, artifacts_dir: str):
        self.model = joblib.load(os.path.join(artifacts_dir, "coarse_model.joblib"))
        self.label_encoder = joblib.load(os.path.join(artifacts_dir, "coarse_label_encoder.joblib"))
        with open(os.path.join(artifacts_dir, "coarse_feature_columns.json")) as f:
            self.feature_cols: List[str] = json.load(f)

    def _feature_row(self, features: Dict[str, float]) -> pd.DataFrame:
        row = {c: features.get(c, np.nan) for c in self.feature_cols}
        return pd.DataFrame([row], columns=self.feature_cols)

    def predict(self, features: Dict[str, float]) -> Dict:
        X = self._feature_row(features)
        pred_idx = int(self.model.predict(X)[0])
        label = str(self.label_encoder.inverse_transform([pred_idx])[0])

        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = float(proba[pred_idx])

        return {
            "coarse_label": label,
            "confidence": confidence,
            "is_electronic": label == ELECTRONIC_LABEL,
        }
