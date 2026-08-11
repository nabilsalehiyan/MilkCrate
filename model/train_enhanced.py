"""
Enhanced librosa feature extraction + LightGBM training.
Adds mel spectrogram statistics, spectral contrast, and 
segment-based features for much richer representation.
Target: 65-75% accuracy vs current 52%.
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import json
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

AUDIO_EXTENSIONS = {'.mp3', '.flac', '.aiff', '.aif', '.wav', '.m4a'}
SR = 22050
MAX_SECS = 60

FAMILY_DEFAULTS = {
    "Tech_House": "House", "Minimal_deep": "House", "Afro_house": "House",
    "Classic_house": "House", "Melodic_house": "House", "DeepHouse": "House",
    "Progressive_House": "House", "House": "House", "ElectroHouse": "House",
    "FutureHouse": "House", "BigRoom": "House", "Bass_house": "Bass",
    "Techno": "Techno", "HardDance": "Techno",
    "Drum_and_bass": "Bass", "UKG": "Bass", "Dubstep": "Bass", "Breaks": "Bass",
    "Trance": "Trance", "PsyTrance": "Trance",
    "Dance": "Electronic", "Downtempo": "Electronic",
    "HipHop_RnB": "Other", "Rock_Alternative": "Other",
    "Pop_Vocal": "Other", "Jazz_Acoustic_Classical": "Other", "Other": "Other",
}

def load_audio(path):
    try:
        wav, sr = sf.read(str(path), always_2d=True)
        wav = wav.mean(axis=1).astype(np.float32)
        if sr != SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    except Exception:
        wav, _ = librosa.load(str(path), sr=SR, mono=True)
        wav = wav.astype(np.float32)
    max_samples = SR * MAX_SECS
    if len(wav) > max_samples:
        start = min(SR * 30, (len(wav) - max_samples) // 2)
        wav = wav[start:start + max_samples]
    return wav

def stat(x):
    """Return mean, std, min, max, median of array"""
    return [np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x)]

def extract_features(wav):
    feats = []
    
    # 1. MFCCs (40 coefficients × 5 stats = 200 features)
    mfcc = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=40)
    for coef in mfcc:
        feats.extend(stat(coef))
    
    # 2. Delta MFCCs (captures rate of change)
    delta = librosa.feature.delta(mfcc)
    for coef in delta:
        feats.extend(stat(coef))
    
    # 3. Mel spectrogram (128 bands × 3 stats = 384 features)
    mel = librosa.feature.melspectrogram(y=wav, sr=SR, n_mels=128)
    mel_db = librosa.power_to_db(mel)
    for band in mel_db:
        feats.extend([np.mean(band), np.std(band), np.max(band)])
    
    # 4. Chroma (12 × 5 stats = 60 features)
    chroma = librosa.feature.chroma_stft(y=wav, sr=SR)
    for c in chroma:
        feats.extend(stat(c))
    
    # 5. Spectral contrast (7 bands × 5 stats = 35 features)
    contrast = librosa.feature.spectral_contrast(y=wav, sr=SR, n_bands=6)
    for band in contrast:
        feats.extend(stat(band))
    
    # 6. Tonnetz (6 × 5 stats = 30 features)
    harmonic = librosa.effects.harmonic(wav)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=SR)
    for t in tonnetz:
        feats.extend(stat(t))
    
    # 7. Spectral features (5 stats each)
    sc = librosa.feature.spectral_centroid(y=wav, sr=SR)[0]
    feats.extend(stat(sc))
    
    sb = librosa.feature.spectral_bandwidth(y=wav, sr=SR)[0]
    feats.extend(stat(sb))
    
    sr_feat = librosa.feature.spectral_rolloff(y=wav, sr=SR)[0]
    feats.extend(stat(sr_feat))
    
    # 8. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(wav)[0]
    feats.extend(stat(zcr))
    
    # 9. RMS energy
    rms = librosa.feature.rms(y=wav)[0]
    feats.extend(stat(rms))
    
    # 10. BPM + beat strength
    tempo, beats = librosa.beat.beat_track(y=wav, sr=SR)
    feats.append(float(tempo))
    if len(beats) > 1:
        beat_times = librosa.frames_to_time(beats, sr=SR)
        feats.append(float(np.std(np.diff(beat_times))))  # beat regularity
    else:
        feats.append(0.0)
    
    # 11. Harmonic/percussive ratio
    harmonic_wav, percussive_wav = librosa.effects.hpss(wav)
    h_energy = float(np.mean(harmonic_wav ** 2))
    p_energy = float(np.mean(percussive_wav ** 2))
    feats.append(h_energy)
    feats.append(p_energy)
    feats.append(h_energy / (p_energy + 1e-8))
    
    return np.array(feats, dtype=np.float32)

def extract_dataset(data_dir, cache_path):
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    data_dir = Path(data_dir)
    classes = sorted([d.name for d in data_dir.iterdir()
                      if d.is_dir() and not d.name.startswith('.')])
    print(f"[INFO] Classes: {classes}")

    records = []
    for label in classes:
        files = [f for f in (data_dir / label).iterdir()
                 if f.suffix.lower() in AUDIO_EXTENSIONS
                 and not f.name.startswith('.')]
        print(f"\n[INFO] {label}: {len(files)} files")
        ok = 0
        for f in files:
            try:
                wav = load_audio(f)
                feats = extract_features(wav)
                rec = {f"ef{i}": float(v) for i, v in enumerate(feats)}
                rec['label'] = label
                rec['file'] = str(f)
                records.append(rec)
                ok += 1
                print(f"  ✓ {f.name}")
            except Exception as e:
                print(f"  ✗ {f.name}: {e}")
        print(f"  → {ok}/{len(files)} extracted")

    df = pd.DataFrame(records)
    print(f"\n[INFO] Total: {len(df)} tracks, {len([c for c in df.columns if c.startswith('f')])} features")

    if cache_path and len(df) > 0:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print(f"[INFO] Cached -> {cache_path}")

    return df

def train(df, out_dir):
    feature_cols = [c for c in df.columns if c.startswith("ef")]
    X = df[feature_cols].values
    y = df['label'].values

    print(f"\n[INFO] {len(df)} tracks, {len(feature_cols)} features, {len(set(y))} classes")
    print(pd.Series(y).value_counts().to_string())

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=3,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False)])

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    proba = model.predict_proba(X_test)
    top3 = np.argsort(proba, axis=1)[:, -3:]
    top3_acc = np.mean([y_test[i] in top3[i] for i in range(len(y_test))])

    print(f"\n=== Evaluation ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Top-3 Acc: {top3_acc:.4f}")
    print(f"\n=== Per-class report ===")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_,
                                digits=4, zero_division=0))

    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "genre_model.joblib"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))

    with open(os.path.join(out_dir, "feature_columns.json"), 'w') as f:
        json.dump(feature_cols, f)

    family_map = {label: FAMILY_DEFAULTS.get(label, "Unknown") for label in le.classes_}
    with open(os.path.join(out_dir, "family_map.json"), 'w') as f:
        json.dump(family_map, f, indent=2)

    with open(os.path.join(out_dir, "model_type.json"), 'w') as f:
        json.dump({"type": "enhanced_librosa", "n_features": len(feature_cols)}, f)

    print(f"\n[INFO] Saved model -> {out_dir}/genre_model.joblib")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache", default="artifacts/enhanced_features.parquet")
    ap.add_argument("--out-dir", default="artifacts")
    args = ap.parse_args()

    df = extract_dataset(args.data_dir, args.cache)
    train(df, args.out_dir)
