"""
Audio data augmentation for MilkCrate DJ.
Multiplies each class 3-4x using pitch shift, time stretch, and noise.
Saves augmented files alongside originals in classified_music_dataset.
"""

import os
import argparse
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

AUDIO_EXTENSIONS = {'.mp3', '.flac', '.aiff', '.aif', '.wav', '.m4a'}

def load_audio(path, sr=22050, max_secs=60):
    try:
        wav, orig_sr = sf.read(str(path), always_2d=True)
        wav = wav.mean(axis=1).astype(np.float32)
        if orig_sr != sr:
            wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    except Exception:
        wav, _ = librosa.load(str(path), sr=sr, mono=True)
        wav = wav.astype(np.float32)
    max_samples = sr * max_secs
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    return wav, sr

def save_audio(wav, sr, path):
    sf.write(str(path), wav, sr)

def augment_pitch_shift(wav, sr, steps):
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=steps)

def augment_time_stretch(wav, rate):
    return librosa.effects.time_stretch(wav, rate=rate)

def augment_add_noise(wav, noise_level=0.005):
    noise = np.random.randn(len(wav)).astype(np.float32) * noise_level
    return np.clip(wav + noise, -1.0, 1.0)

def augment_dataset(data_dir, dry_run=False):
    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir()
                      if d.is_dir() and not d.name.startswith('.')])
    
    print(f"[INFO] Found {len(classes)} classes")
    print(f"[INFO] Augmentations per track: 3 (pitch -1, pitch +1, time stretch)")
    print()

    total_new = 0
    for label in classes:
        class_dir = data_dir / label
        files = [f for f in class_dir.iterdir()
                 if f.suffix.lower() in AUDIO_EXTENSIONS
                 and not f.name.startswith('.')
                 and '_aug_' not in f.name]  # skip already augmented
        
        existing_aug = [f for f in class_dir.iterdir() if '_aug_' in f.name]
        print(f"[INFO] {label}: {len(files)} originals, {len(existing_aug)} existing augmented")
        
        new_count = 0
        for f in files:
            try:
                wav, sr = load_audio(f)
                stem = f.stem
                
                # Augmentation 1: pitch shift down 1 semitone
                out1 = class_dir / f"{stem}_aug_pitch_down.wav"
                if not out1.exists():
                    aug1 = augment_pitch_shift(wav, sr, -1.0)
                    if not dry_run:
                        save_audio(aug1, sr, out1)
                    new_count += 1

                # Augmentation 2: pitch shift up 1 semitone  
                out2 = class_dir / f"{stem}_aug_pitch_up.wav"
                if not out2.exists():
                    aug2 = augment_pitch_shift(wav, sr, 1.0)
                    if not dry_run:
                        save_audio(aug2, sr, out2)
                    new_count += 1

                # Augmentation 3: slight time stretch (5% faster)
                out3 = class_dir / f"{stem}_aug_stretch.wav"
                if not out3.exists():
                    aug3 = augment_time_stretch(wav, 1.05)
                    if not dry_run:
                        save_audio(aug3, sr, out3)
                    new_count += 1

                print(f"  ✓ {f.name}")

            except Exception as e:
                print(f"  ✗ {f.name}: {e}")

        print(f"  → {new_count} new augmented files for {label}")
        total_new += new_count

    print(f"\n[INFO] Total new augmented files: {total_new}")
    print(f"[INFO] Dataset size: {sum(len(files) for files in [[f for f in (data_dir/l).iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS and not f.name.startswith('.')] for l in classes])} tracks")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--dry-run", action="store_true", help="Show what would be created without saving")
    args = ap.parse_args()
    augment_dataset(args.data_dir, dry_run=args.dry_run)
