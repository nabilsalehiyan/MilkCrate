"""
scanner.py
==========
Walks a directory (e.g. a mounted USB drive or music library folder),
finds audio files, extracts features (using TrackCache to skip unchanged
files), classifies genre/family, and flags duplicates.

This module exposes a generator-based `scan_library()` so a GUI can show
live progress without blocking.
"""

from __future__ import annotations

import os
from typing import Dict, Iterator, List, Optional

from feature_extract import extract_features_from_file, DEFAULT_SR, DEFAULT_MAX_SECS
from cache_db import TrackCache
from classifier import TwoStageClassifier
from tag_reader import read_tags
from duplicate_detect import (
    file_content_hash,
    audio_fingerprint,
    fingerprint_to_str,
    fingerprint_from_str,
    find_exact_duplicates,
    find_near_duplicates,
)


AUDIO_EXTENSIONS = {".mp3", ".flac", ".aiff", ".aif", ".wav", ".m4a", ".aac", ".ogg", ".alac"}


def find_audio_files(root: str) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            # Skip macOS AppleDouble resource-fork files (e.g. "._Track.flac")
            # and other hidden/dotfiles — these aren't real audio and exist
            # alongside every real file on non-HFS+ (USB/exFAT) drives.
            if fn.startswith("."):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext in AUDIO_EXTENSIONS:
                paths.append(os.path.join(dirpath, fn))
    return paths


# Maps common store/metadata genre spellings to our canonical labels.
# Comparison is done on a lowercase, alphanumeric-only basis first, then
# via this alias table for names that differ structurally.
_GENRE_ALIASES = {
    "techhouse": "Tech_House",
    "tech house": "Tech_House",
    "minimal": "Minimal_deep",
    "minimaldeep": "Minimal_deep",
    "minimal deep tech": "Minimal_deep",
    "deeptech": "Minimal_deep",
    "afrohouse": "Afro_house",
    "afro house": "Afro_house",
    "classichouse": "Classic_house",
    "housemusic": "Classic_house",
    "house": "Classic_house",
    "melodichouse": "Melodic_house",
    "melodichousetechno": "Melodic_house",
    "melodic house & techno": "Melodic_house",
    "progressivehouse": "Melodic_house",
    "basshouse": "Bass_house",
    "bass house": "Bass_house",
    "ukg": "UKG",
    "ukgarage": "UKG",
    "garage": "UKG",
    "2step": "UKG",
    "drumandbass": "Drum_and_bass",
    "drumbass": "Drum_and_bass",
    "dnb": "Drum_and_bass",
    "jungle": "Drum_and_bass",
    "dubstep": "Dubstep",
    "riddim": "Dubstep",
    "techno": "Techno",
    "technopeaktimedriving": "Techno",
    "techno (peak time / driving)": "Techno",
    "hardtechno": "Techno",
    "trance": "Trance",
    "psytrance": "Trance",
}


def _norm_genre(s: str) -> str:
    """lowercase + strip all non-alphanumerics for fuzzy matching"""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def canonicalize_tag_genre(tag_genre: str) -> Optional[str]:
    """
    Map a raw metadata genre string to one of our canonical labels.
    Returns None if we can't map it (unknown/garbage tag).
    """
    if not tag_genre:
        return None
    raw = tag_genre.strip()
    n = _norm_genre(raw)
    if not n:
        return None
    if n in _GENRE_ALIASES:
        return _GENRE_ALIASES[n]
    # exact match against canonical labels themselves (Tech_House etc.)
    canon = {_norm_genre(k): k for k in set(_GENRE_ALIASES.values())}
    return canon.get(n)


class ScanResult:
    def __init__(self, path: str):
        self.path = path
        self.subgenre: Optional[str] = None
        self.family: Optional[str] = None
        self.top_k: Optional[list] = None
        self.bpm: Optional[float] = None
        self.bpm_source: Optional[str] = None
        self.duration_sec: Optional[float] = None
        self.stage: Optional[str] = None
        self.error: Optional[str] = None
        self.features: Optional[dict] = None  # populated for feedback logging
        self.tag_genre: Optional[str] = None      # genre from file metadata, if any
        self.tag_match: Optional[bool] = None     # True=agrees with prediction, False=conflict, None=no tag


def scan_library(
    root: str,
    cache: TrackCache,
    classifier: "TwoStageClassifier",
    sr: int = DEFAULT_SR,
    max_secs: int = DEFAULT_MAX_SECS,
    compute_content_hash: bool = True,
    stop_event=None,
) -> Iterator[ScanResult]:
    """
    Yields a ScanResult for each audio file found, one at a time, so a
    GUI can update progress incrementally.

    If `stop_event` (a threading.Event) is set between files, the scan
    stops early — already-cached results remain cached for next time.
    """
    paths = find_audio_files(root)

    for path in paths:
        if stop_event is not None and stop_event.is_set():
            return

        result = ScanResult(path)
        try:
            cached = cache.get(path)

            if cached and cached["features"] is not None:
                features = cached["features"]
                pred = {
                    "subgenre": cached["pred_subgenre"],
                    "family": cached["pred_family"],
                    "top_k": cached["pred_topk"],
                    "bpm": cached["bpm"],
                    "stage": "cached",
                }
                duration = cached["duration_sec"]
                content_hash = cached["content_hash"]
                audio_hash = cached["audio_hash"]
            else:
                features = extract_features_from_file(path, sr=sr, max_secs=max_secs)
                pred = classifier.predict(features, top_k=3)
                duration = features.get("_duration_analyzed_sec")

                content_hash = file_content_hash(path) if compute_content_hash else None
                audio_hash = fingerprint_to_str(audio_fingerprint(features))

                cache.put(
                    path,
                    features=features,
                    pred_subgenre=pred["subgenre"],
                    pred_family=pred["family"],
                    pred_topk=pred["top_k"],
                    bpm=pred["bpm"],
                    duration_sec=duration,
                    content_hash=content_hash,
                    audio_hash=audio_hash,
                )

            # Read embedded genre tag and compare with the model prediction.
            # Match -> we keep our (agreeing) prediction and mark agreement.
            # Mismatch -> both are surfaced in the UI in separate columns.
            try:
                raw_tag = read_tags(path).get("genre")
            except Exception:
                raw_tag = None
            if raw_tag:
                result.tag_genre = raw_tag
                mapped = canonicalize_tag_genre(raw_tag)
                if mapped is not None:
                    result.tag_match = (mapped == pred["subgenre"])
                else:
                    result.tag_match = None  # unmappable tag: show it, no verdict

            result.subgenre = pred["subgenre"]
            result.family = pred["family"]
            result.top_k = pred["top_k"]
            result.bpm = pred["bpm"]
            result.bpm_source = features.get("_bpm_source", "unknown")
            result.duration_sec = duration
            result.stage = pred.get("stage")
            result.features = features

        except Exception as e:
            result.error = str(e)

        yield result


def detect_duplicates(root: str, cache: TrackCache) -> Dict[str, List[List[str]]]:
    """
    After a scan has populated the cache, find exact and near duplicates
    among files under `root`.
    """
    paths = find_audio_files(root)

    file_hashes: Dict[str, str] = {}
    fingerprints: Dict[str, "np.ndarray"] = {}

    for path in paths:
        cached = cache.get(path)
        if not cached:
            continue
        if cached.get("content_hash"):
            file_hashes[path] = cached["content_hash"]
        if cached.get("audio_hash"):
            fingerprints[path] = fingerprint_from_str(cached["audio_hash"])

    exact = find_exact_duplicates(file_hashes)

    # exclude exact-dup paths from near-dup search (already grouped)
    exact_paths = {p for group in exact for p in group}
    near_input = {p: v for p, v in fingerprints.items() if p not in exact_paths}
    near = find_near_duplicates(near_input)

    return {"exact": exact, "near": near}
