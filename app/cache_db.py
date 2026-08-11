"""
cache_db.py
===========
SQLite-backed cache so re-scanning a library doesn't re-run expensive
audio feature extraction on files that haven't changed.

Cache key = file fingerprint (path + size + mtime). If a file is moved
but unchanged in content, size+mtime will differ, so it'll be re-analyzed —
this is a deliberate simplicity tradeoff (fast, no audio hashing needed
for the cache itself; see duplicate_detect.py for content-based hashing).
"""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Optional, Dict, Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS track_cache (
    fingerprint   TEXT PRIMARY KEY,
    path          TEXT NOT NULL,
    size          INTEGER NOT NULL,
    mtime         REAL NOT NULL,
    features_json TEXT,
    pred_subgenre TEXT,
    pred_family   TEXT,
    pred_topk_json TEXT,
    bpm           REAL,
    duration_sec  REAL,
    content_hash  TEXT,       -- for exact-duplicate detection (cheap hash)
    audio_hash    TEXT,       -- for near-duplicate / acoustic fingerprint
    updated_at    REAL
);

CREATE INDEX IF NOT EXISTS idx_content_hash ON track_cache(content_hash);
CREATE INDEX IF NOT EXISTS idx_audio_hash ON track_cache(audio_hash);
"""


class TrackCache:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    @staticmethod
    def fingerprint(path: str) -> str:
        st = os.stat(path)
        return f"{path}|{st.st_size}|{st.st_mtime}"

    def get(self, path: str) -> Optional[Dict[str, Any]]:
        fp = self.fingerprint(path)
        cur = self.conn.execute(
            "SELECT features_json, pred_subgenre, pred_family, pred_topk_json, bpm, duration_sec, "
            "content_hash, audio_hash FROM track_cache WHERE fingerprint = ?",
            (fp,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "features": json.loads(row[0]) if row[0] else None,
            "pred_subgenre": row[1],
            "pred_family": row[2],
            "pred_topk": json.loads(row[3]) if row[3] else None,
            "bpm": row[4],
            "duration_sec": row[5],
            "content_hash": row[6],
            "audio_hash": row[7],
        }

    def put(
        self,
        path: str,
        features: Optional[Dict[str, float]] = None,
        pred_subgenre: Optional[str] = None,
        pred_family: Optional[str] = None,
        pred_topk: Optional[list] = None,
        bpm: Optional[float] = None,
        duration_sec: Optional[float] = None,
        content_hash: Optional[str] = None,
        audio_hash: Optional[str] = None,
    ):
        import time
        fp = self.fingerprint(path)
        st = os.stat(path)
        self.conn.execute(
            """
            INSERT INTO track_cache
                (fingerprint, path, size, mtime, features_json, pred_subgenre,
                 pred_family, pred_topk_json, bpm, duration_sec, content_hash, audio_hash, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fingerprint) DO UPDATE SET
                features_json=excluded.features_json,
                pred_subgenre=excluded.pred_subgenre,
                pred_family=excluded.pred_family,
                pred_topk_json=excluded.pred_topk_json,
                bpm=excluded.bpm,
                duration_sec=excluded.duration_sec,
                content_hash=excluded.content_hash,
                audio_hash=excluded.audio_hash,
                updated_at=excluded.updated_at
            """,
            (
                fp, path, st.st_size, st.st_mtime,
                json.dumps(features) if features is not None else None,
                pred_subgenre, pred_family,
                json.dumps(pred_topk) if pred_topk is not None else None,
                bpm, duration_sec, content_hash, audio_hash,
                time.time(),
            ),
        )
        self.conn.commit()

    def find_by_content_hash(self, content_hash: str):
        cur = self.conn.execute(
            "SELECT path FROM track_cache WHERE content_hash = ?", (content_hash,)
        )
        return [r[0] for r in cur.fetchall()]

    def all_audio_hashes(self):
        """Return list of (path, audio_hash) for near-duplicate comparison."""
        cur = self.conn.execute(
            "SELECT path, audio_hash FROM track_cache WHERE audio_hash IS NOT NULL"
        )
        return cur.fetchall()

    def close(self):
        self.conn.close()
