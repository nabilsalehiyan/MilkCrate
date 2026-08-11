"""
MilkCrate Feedback Backend
Collects genre corrections + feature vectors from users,
serves updated models back to the app.

Deploy on Hetzner CX22 (~€4.55/mo)
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import asyncpg

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://milkcrate:milkcrate@localhost/milkcrate")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/milkcrate/models")
API_KEYS = set(os.environ.get("API_KEYS", "beta-key-1").split(","))

app = FastAPI(title="MilkCrate Backend", version="1.0.0")

# ---------- Models ----------

class FeedbackSubmission(BaseModel):
    features: List[float] = Field(..., min_length=900, max_length=1000)
    predicted_label: str
    corrected_label: str
    confidence: Optional[float] = None
    app_version: Optional[str] = None
    model_version: Optional[str] = None

class FeedbackBatch(BaseModel):
    corrections: List[FeedbackSubmission]

# ---------- Auth ----------

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

# ---------- DB ----------

pool = None

@app.on_event("startup")
async def startup():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id SERIAL PRIMARY KEY,
                submitted_at TIMESTAMPTZ DEFAULT NOW(),
                api_key_hash TEXT,
                features JSONB NOT NULL,
                predicted_label TEXT NOT NULL,
                corrected_label TEXT NOT NULL,
                confidence REAL,
                app_version TEXT,
                model_version TEXT,
                used_in_training BOOLEAN DEFAULT FALSE
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                id SERIAL PRIMARY KEY,
                version TEXT UNIQUE NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                accuracy REAL,
                n_training_samples INT,
                filename TEXT NOT NULL,
                is_current BOOLEAN DEFAULT FALSE
            )
        """)

@app.on_event("shutdown")
async def shutdown():
    await pool.close()

# ---------- Endpoints ----------

@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.post("/api/v1/feedback")
async def submit_feedback(batch: FeedbackBatch, api_key: str = Depends(verify_api_key)):
    """Receive a batch of corrections from a user's app."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    async with pool.acquire() as conn:
        count = 0
        for c in batch.corrections:
            await conn.execute(
                """INSERT INTO feedback
                   (api_key_hash, features, predicted_label, corrected_label,
                    confidence, app_version, model_version)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                key_hash, json.dumps(c.features), c.predicted_label,
                c.corrected_label, c.confidence, c.app_version, c.model_version
            )
            count += 1
    return {"received": count}

@app.get("/api/v1/model/latest")
async def latest_model_info():
    """Check what the current model version is."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT version, created_at, accuracy, n_training_samples FROM model_versions WHERE is_current = TRUE"
        )
    if not row:
        raise HTTPException(status_code=404, detail="No model published yet")
    return dict(row)

@app.get("/api/v1/model/download/{version}")
async def download_model(version: str):
    """Download a model bundle (zip of artifacts)."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT filename FROM model_versions WHERE version = $1", version
        )
    if not row:
        raise HTTPException(status_code=404, detail="Version not found")
    path = os.path.join(MODEL_DIR, row["filename"])
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Model file missing")
    return FileResponse(path, filename=row["filename"])

@app.get("/api/v1/stats")
async def stats(api_key: str = Depends(verify_api_key)):
    """Dashboard stats — total corrections, breakdown by label."""
    async with pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM feedback")
        unused = await conn.fetchval("SELECT COUNT(*) FROM feedback WHERE NOT used_in_training")
        by_label = await conn.fetch(
            """SELECT corrected_label, COUNT(*) as n FROM feedback
               GROUP BY corrected_label ORDER BY n DESC"""
        )
    return {
        "total_corrections": total,
        "pending_for_training": unused,
        "by_label": {r["corrected_label"]: r["n"] for r in by_label},
    }
