"""
Server-side retraining script.
Run weekly via cron: pulls unused feedback from DB,
combines with base features, retrains, publishes new model version.

Usage: python3 retrain_server.py
"""

import os
import json
import zipfile
import asyncio
from datetime import datetime

import asyncpg
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import lightgbm as lgb

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://milkcrate:milkcrate@localhost/milkcrate")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/milkcrate/models")
BASE_FEATURES = os.environ.get("BASE_FEATURES", "/opt/milkcrate/base_features.parquet")
MIN_NEW_SAMPLES = 20  # don't retrain for less than this

async def main():
    conn = await asyncpg.connect(DATABASE_URL)

    rows = await conn.fetch("SELECT id, features, corrected_label FROM feedback WHERE NOT used_in_training")
    if len(rows) < MIN_NEW_SAMPLES:
        print(f"Only {len(rows)} new corrections — need {MIN_NEW_SAMPLES}. Skipping.")
        await conn.close()
        return

    print(f"[INFO] {len(rows)} new corrections to fold in")

    # Load base training features
    base_df = pd.read_parquet(BASE_FEATURES)
    feature_cols = [c for c in base_df.columns if c.startswith("ef")]

    # Convert feedback to rows
    fb_records = []
    for r in rows:
        feats = json.loads(r["features"])
        rec = {f"ef{i}": v for i, v in enumerate(feats)}
        rec["label"] = r["corrected_label"]
        fb_records.append(rec)
    fb_df = pd.DataFrame(fb_records)

    combined = pd.concat([base_df[feature_cols + ["label"]], fb_df[feature_cols + ["label"]]], ignore_index=True)
    print(f"[INFO] Training on {len(combined)} total samples")

    X = combined[feature_cols].values
    y = combined["label"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                                min_child_samples=3, subsample=0.8, colsample_bytree=0.8,
                                class_weight='balanced', random_state=42, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False)])

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[INFO] New model accuracy: {acc:.4f}")

    # Save versioned bundle
    version = datetime.utcnow().strftime("%Y%m%d-%H%M")
    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle_name = f"model-{version}.zip"
    bundle_path = os.path.join(MODEL_DIR, bundle_name)

    tmp = f"/tmp/model-{version}"
    os.makedirs(tmp, exist_ok=True)
    joblib.dump(model, f"{tmp}/genre_model.joblib")
    joblib.dump(le, f"{tmp}/label_encoder.joblib")
    joblib.dump(scaler, f"{tmp}/scaler.joblib")
    with open(f"{tmp}/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in os.listdir(tmp):
            z.write(os.path.join(tmp, fn), fn)

    # Publish
    await conn.execute("UPDATE model_versions SET is_current = FALSE")
    await conn.execute(
        """INSERT INTO model_versions (version, accuracy, n_training_samples, filename, is_current)
           VALUES ($1, $2, $3, $4, TRUE)""",
        version, acc, len(combined), bundle_name
    )
    await conn.execute("UPDATE feedback SET used_in_training = TRUE WHERE NOT used_in_training")
    await conn.close()
    print(f"[INFO] Published model {version}")

if __name__ == "__main__":
    asyncio.run(main())
