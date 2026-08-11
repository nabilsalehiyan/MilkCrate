"""
duplicate_detect.py
====================
Two layers of duplicate detection:

1. Exact duplicates: SHA-1 hash of file content. Catches byte-identical
   copies (same file in multiple folders) — instant, no audio decode.

2. Near duplicates: same track ripped/encoded differently (MP3 vs FLAC,
   different bitrates, slightly different lengths from a re-rip). We
   compute a compact "audio fingerprint" from the chroma feature already
   extracted during classification (chromavector1m..12m + bpm), and group
   tracks whose fingerprints are within a small distance of each other.

Near-duplicates are only *flagged* for the user to review/merge — never
auto-deleted.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Tuple
import numpy as np


CHUNK_SIZE = 1024 * 1024  # 1MB


def file_content_hash(path: str) -> str:
    """Fast SHA-1 of file bytes (streamed)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


CHROMA_KEYS = [f"chromavector{i}m" for i in range(1, 13)]


def audio_fingerprint(features: Dict[str, float]) -> np.ndarray:
    """
    Build a compact numeric fingerprint vector from extracted features —
    chroma profile (pitch content, robust to encoding) + BPM + spectral
    centroid. Tracks that are the same song in different formats will have
    near-identical chroma + BPM regardless of bitrate/codec.
    """
    vec = []
    for k in CHROMA_KEYS:
        vec.append(features.get(k, 0.0) or 0.0)
    vec.append((features.get("bpm", 0.0) or 0.0) / 200.0)  # normalize roughly to 0-1
    vec.append((features.get("spectralcentroidm", 0.0) or 0.0))
    return np.array(vec, dtype=np.float64)


def fingerprint_to_str(vec: np.ndarray) -> str:
    """Serialize fingerprint for storage."""
    return ",".join(f"{x:.5f}" for x in vec)


def fingerprint_from_str(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split(",")], dtype=np.float64)


def find_exact_duplicates(file_hashes: Dict[str, str]) -> List[List[str]]:
    """
    file_hashes: {path: content_hash}
    Returns groups of paths that share the same content hash (size > 1).
    """
    groups: Dict[str, List[str]] = {}
    for path, h in file_hashes.items():
        groups.setdefault(h, []).append(path)
    return [g for g in groups.values() if len(g) > 1]


def find_near_duplicates(
    fingerprints: Dict[str, np.ndarray],
    threshold: float = 0.08,
) -> List[List[str]]:
    """
    fingerprints: {path: fingerprint_vector}
    Groups tracks whose fingerprint Euclidean distance < threshold.
    O(n^2) — fine for library sizes up to a few thousand tracks.
    Returns groups of 2+ paths considered likely the same track.
    """
    paths = list(fingerprints.keys())
    n = len(paths)
    visited = set()
    groups: List[List[str]] = []

    for i in range(n):
        if paths[i] in visited:
            continue
        group = [paths[i]]
        for j in range(i + 1, n):
            if paths[j] in visited:
                continue
            dist = np.linalg.norm(fingerprints[paths[i]] - fingerprints[paths[j]])
            if dist < threshold:
                group.append(paths[j])
                visited.add(paths[j])
        if len(group) > 1:
            visited.add(paths[i])
            groups.append(group)

    return groups
