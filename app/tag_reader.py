"""
tag_reader.py
=============
Reads embedded metadata (BPM, title, artist, genre tag) from audio files
using mutagen. Most DJ-pool tracks (Beatport, etc.) already have accurate
BPM tagged via Mixed In Key / Rekordbox / the original release — far more
reliable than estimating tempo from 90 seconds of audio, especially for
tracks with syncopated percussion where tempogram-based detection can lock
onto a 3:4 or 4:3 polyrhythm ratio instead of the true tempo.

This is used as the PRIMARY BPM source when available, with our own
librosa-based tempo estimate (feature_extract._tempo_histogram_feats) as
fallback only for untagged files.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import mutagen


def read_tags(path: str) -> Dict[str, Any]:
    """
    Returns a dict with whatever of these keys could be read:
        bpm    (float)
        title  (str)
        artist (str)
        genre  (str)
    Missing/unreadable fields are simply absent from the dict.
    """
    result: Dict[str, Any] = {}
    try:
        audio = mutagen.File(path, easy=True)
    except Exception:
        return result

    if audio is None:
        return result

    # mutagen's "easy" tags normalize keys across formats (mp3/flac/m4a/etc.)
    tags = getattr(audio, "tags", None) or {}

    def _first(key: str) -> Optional[str]:
        val = tags.get(key)
        if val:
            if isinstance(val, list):
                return str(val[0])
            return str(val)
        return None

    bpm_str = _first("bpm")
    if bpm_str:
        try:
            # Some tools store as "123.00 BPM" or "123" or "123.0"
            cleaned = bpm_str.strip().split()[0]
            bpm_val = float(cleaned)
            if 30.0 <= bpm_val <= 300.0:  # sanity check
                result["bpm"] = bpm_val
        except (ValueError, IndexError):
            pass

    title = _first("title")
    if title:
        result["title"] = title

    artist = _first("artist")
    if artist:
        result["artist"] = artist

    genre = _first("genre")
    if genre:
        result["genre"] = genre

    return result


def get_tagged_bpm(path: str) -> Optional[float]:
    """Convenience: return just the BPM tag if present and valid, else None."""
    return read_tags(path).get("bpm")
