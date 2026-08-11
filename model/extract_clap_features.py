"""
CLAP-based feature extractor for MilkCrate DJ.
Uses soundfile + librosa for audio loading (handles flac/aiff/wav/mp3).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import librosa
from pathlib import Path
from transformers import ClapModel, ClapProcessor

AUDIO_EXTENSIONS = {'.mp3', '.flac', '.aiff', '.aif', '.wav', '.m4a', '.ogg'}
MODEL_ID = "laion/larger_clap_music"
TARGET_SR = 48000

def load_clap(device):
    print(f"[INFO] Loading CLAP model...")
    model = ClapModel.from_pretrained(MODEL_ID)
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model.eval()
    model.to(device)
    print(f"[INFO] CLAP ready on {device}")
    return model, processor

def load_audio(path, max_secs=60):
    try:
        wav, sr = sf.read(str(path), always_2d=True)
        wav = wav.mean(axis=1).astype(np.float32)
    except Exception:
        # fallback to librosa for mp3/m4a
        wav, sr = librosa.load(str(path), sr=None, mono=True)
        wav = wav.astype(np.float32)

    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)

    max_samples = TARGET_SR * max_secs
    if len(wav) > max_samples:
        start = (len(wav) - max_samples) // 2
        wav = wav[start:start + max_samples]

    return wav

def get_embedding(model, processor, audio, device):
    inputs = processor(audios=audio, sampling_rate=TARGET_SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_audio_features(**inputs)
    return emb.cpu().numpy().squeeze()

def extract_dataset(data_dir, cache_path, device, max_secs=60):
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    model, processor = load_clap(device)
    records = []
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir()
                      if d.is_dir() and not d.name.startswith('.')])
    print(f"[INFO] Classes: {classes}")

    for label in classes:
        files = [f for f in (data_dir / label).iterdir()
                 if f.suffix.lower() in AUDIO_EXTENSIONS
                 and not f.name.startswith('.')]
        print(f"\n[INFO] {label}: {len(files)} files")
        ok = 0
        for f in files:
            try:
                audio = load_audio(f, max_secs=max_secs)
                emb = get_embedding(model, processor, audio, device)
                rec = {f"clap_{i}": float(v) for i, v in enumerate(emb)}
                rec['label'] = label
                rec['file'] = str(f)
                records.append(rec)
                ok += 1
                print(f"  ✓ {f.name}")
            except Exception as e:
                print(f"  ✗ {f.name}: {e}")
        print(f"  → {ok}/{len(files)} extracted")

    df = pd.DataFrame(records)
    print(f"\n[INFO] Total: {len(df)} tracks extracted")

    if cache_path and len(df) > 0:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"[INFO] Cached -> {cache_path}")

    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache", default="artifacts/clap_features.parquet")
    ap.add_argument("--max-secs", type=int, default=60)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    df = extract_dataset(args.data_dir, args.cache, device, args.max_secs)
    print(df['label'].value_counts())
