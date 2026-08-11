"""
feature_extract.py
===================
Shared audio feature extraction used both for:
  - normalizing the BeatsDataset CSV columns into a canonical feature schema
  - extracting the same features live from a user's audio file (librosa)

Canonical feature names follow the lowercase scheme used in the reference
MilkCrate predict.py (e.g. 'zcrm', 'mfccs1m', 'chromavector1m', 'bpm', ...),
which is also what the trained model's feature_names_in_ will use after
we normalize the training CSV columns the same way.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple, List

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_SR = 22050
DEFAULT_MAX_SECS = 90  # analyze first 90s — good speed/accuracy tradeoff for genre


# ---------------------------------------------------------------------------
# CSV column name normalization (for beatsdataset_full.csv -> canonical names)
# ---------------------------------------------------------------------------
# The dataset uses headers like "1-ZCRm", "9-MFCCs1m", "22-ChromaVector1m",
# "69-BPM", "77-danceability", "78-beats_loudness.mean", etc.
# We strip the leading "N-" prefix and lowercase to get canonical names like
# "zcrm", "mfccs1m", "chromavector1m", "bpm", "danceability", "beats_loudness.mean".

_PREFIX_RE = re.compile(r"^\d+-")


def normalize_csv_column(col: str) -> str:
    """Convert a beatsdataset_full.csv column name to canonical lowercase form."""
    stripped = _PREFIX_RE.sub("", col)
    return stripped.lower()


def normalize_csv_columns(columns: List[str]) -> List[str]:
    return [normalize_csv_column(c) for c in columns]


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def load_audio_any(path: str, target_sr: int = DEFAULT_SR, max_secs: int = DEFAULT_MAX_SECS) -> Tuple[np.ndarray, int]:
    """Load audio (mp3/flac/aiff/wav/m4a/...) as mono, resampled, truncated to max_secs."""
    y, sr = librosa.load(path, sr=target_sr, mono=True, duration=max_secs)
    return y, sr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_mean_std(X: np.ndarray) -> Tuple[float, float]:
    if X.size == 0 or np.all(~np.isfinite(X)):
        return float("nan"), float("nan")
    return float(np.nanmean(X)), float(np.nanstd(X))


def _beat_boundaries(y: np.ndarray, sr: int, hop: int, n_frames: int) -> List[Tuple[int, int]]:
    try:
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, units="frames")
        beats = np.asarray(beats, int)
        if beats.size >= 2:
            starts = beats
            ends = np.r_[beats[1:], n_frames]
            pairs = [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]
            if pairs:
                return pairs
    except Exception:
        pass
    step = max(1, int(round(0.5 * sr / hop)))
    starts = np.arange(0, n_frames, step, dtype=int)
    ends = np.r_[starts[1:], n_frames]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def _compute_beats_loudness_band_ratio(y: np.ndarray, sr: int, S_power: np.ndarray, hop: int) -> Dict[str, float]:
    n_fft = 2048
    n_frames = S_power.shape[1]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    nyq = sr / 2.0
    f_min = 20.0
    edges = np.geomspace(f_min, nyq, num=13)
    band_bins = []
    for j in range(12):
        lo, hi = edges[j], edges[j + 1]
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if idx.size == 0:
            idx = np.array([np.argmin(np.abs(freqs - (lo + hi) / 2.0))])
        band_bins.append(idx)

    beats = _beat_boundaries(y, sr, hop, n_frames)
    eps = 1e-10
    full_power = S_power.sum(axis=0) + eps

    ratios = [[] for _ in range(12)]
    for s, e in beats:
        s = max(0, min(s, n_frames - 1))
        e = max(s + 1, min(e, n_frames))
        denom = float(full_power[s:e].mean())
        if not np.isfinite(denom) or denom <= 0:
            continue
        for j in range(12):
            band_pow = S_power[band_bins[j], s:e].mean()
            ratios[j].append(float(band_pow / denom))

    feats = {}
    for j in range(12):
        arr = np.asarray(ratios[j], float)
        m, s = _safe_mean_std(arr) if arr.size else (float("nan"), float("nan"))
        feats[f"beats_loudness_band_ratio.mean{j + 1}"] = m
        feats[f"beats_loudness_band_ratio.stdev{j + 1}"] = s
    return feats


def _compute_beats_loudness_stats(y: np.ndarray, sr: int, hop: int, n_frames: int) -> Dict[str, float]:
    beats = _beat_boundaries(y, sr, hop, n_frames)
    n_fft = 2048
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).squeeze()
    vals = []
    for s, e in beats:
        s = max(0, min(s, len(rms) - 1))
        e = max(s + 1, min(e, len(rms)))
        vals.append(float(np.nanmean(rms[s:e])))
    arr = np.asarray(vals, float)
    m, s = _safe_mean_std(arr)
    return {"beats_loudness.mean": m, "beats_loudness.stdev": s}


def _tempo_histogram_feats(y: np.ndarray, sr: int, hop: int) -> Dict[str, float]:
    keys = [
        "bpm", "bpmconf", "bpm_histogram_first_peak_bpm", "bpm_histogram_first_peak_weight",
        "bpm_histogram_second_peak_bpm", "bpm_histogram_second_peak_weight",
        "bpm_histogram_second_peak_spread", "danceability",
    ]
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if oenv.size == 0:
        return {k: float("nan") for k in keys}

    tg = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop)
    tg_mean = tg.mean(axis=1)
    tempi = librosa.tempo_frequencies(tg_mean.shape[0], sr=sr, hop_length=hop)

    mask = (tempi >= 60) & (tempi <= 200)
    if not np.any(mask):
        mask = (tempi >= 30) & (tempi <= 240)
    t_vals = tempi[mask]
    h_vals = tg_mean[mask]
    if h_vals.size == 0 or np.all(~np.isfinite(h_vals)):
        return {k: float("nan") for k in keys}

    h_vals = np.maximum(h_vals, 0)
    h_norm = h_vals / float(h_vals.max()) if float(h_vals.max()) > 0 else h_vals

    order = np.argsort(h_vals)[::-1]
    i1 = int(order[0]); bpm1 = float(t_vals[i1]); w1 = float(h_norm[i1])
    i2 = int(order[1]) if order.size > 1 else i1
    bpm2 = float(t_vals[i2]); w2 = float(h_norm[i2])

    # --- Octave-error correction ---
    # Onset-strength tempograms frequently lock onto the half-tempo for
    # four-on-the-floor house/techno (strong off-beat energy makes the
    # autocorrelation peak at half the true BPM). If the top peak is in
    # the "half-time slow genre" range (55-95 BPM) but its double also
    # has substantial energy in the histogram, prefer the doubled value —
    # house/techno/trance are virtually always 110-150 BPM, while genuine
    # 55-95 BPM tracks (DnB at half-time, hip-hop, dub/reggae) are rarer
    # in a typical electronic library.
    if 55.0 <= bpm1 <= 95.0:
        doubled = bpm1 * 2.0
        # find the closest histogram bin to the doubled value
        idx_double = int(np.argmin(np.abs(t_vals - doubled)))
        w_double = float(h_norm[idx_double])
        # if the doubled tempo has at least ~35% of the peak's weight,
        # treat it as the true tempo (peak was a half-tempo subharmonic)
        if w_double >= 0.35 * w1:
            bpm2 = bpm1
            w2 = w1
            bpm1 = float(t_vals[idx_double])
            w1 = w_double

    lo = max(0, i2 - 3); hi = min(len(t_vals), i2 + 4)
    t_win = t_vals[lo:hi]; w_win = h_norm[lo:hi]
    if w_win.sum() > 0:
        mu = np.sum(t_win * w_win) / np.sum(w_win)
        spread = float(np.sqrt(np.sum(w_win * (t_win - mu) ** 2) / np.sum(w_win)))
    else:
        spread = float("nan")

    return {
        "bpm": bpm1,
        "bpmconf": w1,
        "bpm_histogram_first_peak_bpm": bpm1,
        "bpm_histogram_first_peak_weight": w1,
        "bpm_histogram_second_peak_bpm": bpm2,
        "bpm_histogram_second_peak_weight": w2,
        "bpm_histogram_second_peak_spread": spread,
        "danceability": w1,
    }


# ---------------------------------------------------------------------------
# Main extraction entry point — returns canonical (lowercase) feature names
# ---------------------------------------------------------------------------
def extract_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Extract the canonical feature set from a mono audio signal."""
    n_fft = 2048
    hop = 512
    feats: Dict[str, float] = {}

    S = np.abs(librosa.stft(y=y, n_fft=n_fft, hop_length=hop)) + 1e-12
    S_power = S * S

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop).squeeze()
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).squeeze()
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop).squeeze()
    sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop).squeeze()
    sroll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop).squeeze()

    # Normalize Hz-scale spectral features by Nyquist frequency (sr/2).
    # The training dataset (BeatsDataset / pyAudioAnalysis-style extraction)
    # stores these as fractions of Nyquist (~0.08-0.37 range), not raw Hz
    # (~1000-5000 range). Without this, these features are ~10,000x outside
    # the training distribution and effectively act as noise to the model.
    nyquist = sr / 2.0
    sc = sc / nyquist
    sbw = sbw / nyquist
    sroll = sroll / nyquist

    P = (S / S.sum(axis=0, keepdims=True)).clip(min=1e-12)
    sent = (-P * np.log2(P)).sum(axis=0)

    Er = (rms ** 2).astype(np.float64)
    Er = Er / (Er.sum() + 1e-12)
    eent = -np.where(Er > 0, Er * np.log2(Er), 0.0)

    dS = np.diff(S, axis=1)
    sflux = np.sqrt((dS * dS).mean(axis=0))
    sflux = np.pad(sflux, (1, 0), mode="constant")

    pairs = [
        ("zcrm", "zcrstd", zcr),
        ("energym", "energystd", rms),
        ("spectralcentroidm", "spectralcentroidstd", sc),
        ("spectralspreadm", "spectralspreadstd", sbw),
        ("spectralrolloffm", "spectralrolloffstd", sroll),
        ("energyentropym", "energyentropystd", eent),
        ("spectralentropym", "spectralentropystd", sent),
        ("spectralfluxm", "spectralfluxstd", sflux),
    ]
    for mean_key, std_key, arr in pairs:
        m, s = _safe_mean_std(arr)
        feats[mean_key] = m
        feats[std_key] = s

    # MFCC + delta-MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)
    for i in range(13):
        m, s = _safe_mean_std(mfcc[i])
        feats[f"mfccs{i+1}m"] = m
        feats[f"mfccs{i+1}std"] = s
    dmfcc = librosa.feature.delta(mfcc, order=1)
    for i in range(13):
        m, s = _safe_mean_std(dmfcc[i])
        feats[f"amfccs{i+1}m"] = m
        feats[f"amfccs{i+1}std"] = s

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    for i in range(12):
        m, s = _safe_mean_std(chroma[i])
        feats[f"chromavector{i+1}m"] = m
        feats[f"chromavector{i+1}std"] = s
    ch_dev = chroma.std(axis=0)
    m, s = _safe_mean_std(ch_dev)
    feats["chromadeviationm"] = m
    feats["chromadeviationstd"] = s

    # Beats-loudness band ratios (24 features)
    feats.update(_compute_beats_loudness_band_ratio(y, sr, S_power=S_power, hop=hop))

    # BPM / tempo histogram / danceability
    feats.update(_tempo_histogram_feats(y, sr, hop))

    # Beats loudness mean/std
    feats.update(_compute_beats_loudness_stats(y, sr, hop, S.shape[1]))

    # Onset rate
    try:
        oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        on_frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop, units="frames")
        dur = len(y) / float(sr) if sr else 0.0
        feats["onset_rate"] = float(len(on_frames) / dur) if dur > 0 else float("nan")
    except Exception:
        feats["onset_rate"] = float("nan")

    return feats


def extract_features_from_file(path: str, sr: int = DEFAULT_SR, max_secs: int = DEFAULT_MAX_SECS) -> Dict[str, float]:
    """
    Convenience: load an audio file and extract canonical features + bpm/duration.

    BPM is taken from the file's embedded tag (Mixed In Key / Rekordbox /
    Beatport-sourced) when available and valid, since this is far more
    reliable than estimating tempo from a 90-second clip — tempogram-based
    detection can lock onto a 3:4 or 4:3 polyrhythm ratio (e.g. reporting
    ~161 BPM for a 125 BPM track with syncopated/shuffled percussion).
    Falls back to the audio-derived estimate (with octave correction) for
    untagged files.
    """
    y, _sr = load_audio_any(path, target_sr=sr, max_secs=max_secs)
    feats = extract_features(y, sr)
    feats["_duration_analyzed_sec"] = len(y) / float(sr) if sr else 0.0

    try:
        from tag_reader import get_tagged_bpm
        tagged_bpm = get_tagged_bpm(path)
        if tagged_bpm is not None:
            feats["bpm"] = tagged_bpm
            feats["bpm_histogram_first_peak_bpm"] = tagged_bpm
            feats["_bpm_source"] = "tag"
        else:
            feats["_bpm_source"] = "detected"
    except Exception:
        feats["_bpm_source"] = "detected"

    # bpmessentia: the BeatsDataset CSV includes a second BPM estimate from
    # the Essentia library (column "71-BPMessentia"), and it's the single
    # most important feature in the trained model (~2x the next feature's
    # importance). Our extraction doesn't run Essentia separately, so alias
    # it to our best BPM estimate (tag if available, else detected) —
    # without this, the model's top feature is NaN on every live prediction.
    feats["bpmessentia"] = feats["bpm"]

    # Enhanced features (used when enhanced model is loaded)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ef = _extract_enhanced(y, sr)
            feats.update(ef)
    except Exception:
        pass

    return feats


def _extract_enhanced(wav: np.ndarray, sr: int) -> dict:
    """Additional rich features for enhanced model."""
    feats = {}
    idx = 0

    def add(arr):
        nonlocal idx
        for v in [np.mean(arr), np.std(arr), np.min(arr), np.max(arr), np.median(arr)]:
            feats[f"ef{idx}"] = float(v)
            idx += 1

    # MFCCs 40 coef x 5 stats
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40)
    for c in mfcc:
        add(c)
    # Delta MFCCs
    delta = librosa.feature.delta(mfcc)
    for c in delta:
        add(c)
    # Mel spectrogram 128 bands x 3 stats
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel)
    for band in mel_db:
        feats[f"ef{idx}"] = float(np.mean(band)); idx += 1
        feats[f"ef{idx}"] = float(np.std(band)); idx += 1
        feats[f"ef{idx}"] = float(np.max(band)); idx += 1
    # Chroma
    chroma = librosa.feature.chroma_stft(y=wav, sr=sr)
    for c in chroma:
        add(c)
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=wav, sr=sr, n_bands=6)
    for c in contrast:
        add(c)
    # Tonnetz
    harmonic = librosa.effects.harmonic(wav)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    for t in tonnetz:
        add(t)
    # Spectral features
    add(librosa.feature.spectral_centroid(y=wav, sr=sr)[0])
    add(librosa.feature.spectral_bandwidth(y=wav, sr=sr)[0])
    add(librosa.feature.spectral_rolloff(y=wav, sr=sr)[0])
    add(librosa.feature.zero_crossing_rate(wav)[0])
    add(librosa.feature.rms(y=wav)[0])
    # BPM features
    tempo, beats = librosa.beat.beat_track(y=wav, sr=sr)
    feats[f"ef{idx}"] = float(tempo); idx += 1
    if len(beats) > 1:
        bt = librosa.frames_to_time(beats, sr=sr)
        feats[f"ef{idx}"] = float(np.std(np.diff(bt)))
    else:
        feats[f"ef{idx}"] = 0.0
    idx += 1
    # Harmonic/percussive
    h, p = librosa.effects.hpss(wav)
    h_e = float(np.mean(h ** 2))
    p_e = float(np.mean(p ** 2))
    feats[f"ef{idx}"] = h_e; idx += 1
    feats[f"ef{idx}"] = p_e; idx += 1
    feats[f"ef{idx}"] = h_e / (p_e + 1e-8); idx += 1

    return feats
