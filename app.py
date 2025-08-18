# app.py — MilkCrate Streamlit app (audio files → features → genre prediction → ZIP by folders)
# - Accepts multiple audio files (mp3/wav/flac/ogg/m4a/aac/wma/aiff, etc.)
# - Extracts features with librosa
# - Uses your sklearn model + LabelEncoder to predict human-readable genres
# - Organizes originals into genre-named folders and offers a ZIP for download

import os
import io
import re
import json
import zipfile
import unicodedata
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Audio
import librosa
import soundfile as sf  # ensures many formats via libsndfile
import audioread        # fallback decoder
from typing import Dict, List, Tuple

# --------------------
# Paths (you can change in the sidebar at runtime)
# --------------------
DEFAULT_MODEL_PATH = "artifacts/beatport201611_hgb.joblib"     # safe-size model you committed (46MB)
DEFAULT_ENCODER_PATH = "artifacts/label_encoder.joblib"        # 23-class encoder you created
TARGET_SR_DEFAULT = 22050

st.set_page_config(page_title="MilkCrate • Audio → Genre Folders", layout="wide")
warnings.filterwarnings("ignore")

# --------------------
# Caching loaders
# --------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model not found at: {path}")
        st.stop()
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_encoder(path: str):
    if not os.path.exists(path):
        st.error(f"Encoder not found at: {path}")
        st.stop()
    return joblib.load(path)

# --------------------
# Utilities
# --------------------
def sanitize_filename(name: str) -> str:
    # Remove path parts & normalize unicode, make filesystem-friendly
    base = os.path.basename(name)
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    base = re.sub(r"[^\w\-.]+", "_", base).strip("._")
    return base or "audio"

def align_columns_to_model(X: pd.DataFrame, model):
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        missing = [c for c in names if c not in X.columns]
        if missing:
            st.warning(f"Input features missing {len(missing)} expected columns. First few: {missing[:10]}")
        X = X.reindex(columns=names)
    return X

def get_display_names(model, encoder):
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None, None
    classes = np.array(classes)
    # Model uses numeric class codes 0..22; map to genre names
    if np.issubdtype(classes.dtype, np.number):
        try:
            names = encoder.inverse_transform(classes.astype(int))
        except Exception:
            names = classes.astype(str)
        return classes, names
    return classes, classes

# --------------------
# Feature extraction
# --------------------
def extract_features_array(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Compute a robust set of features. Names are generic; your model’s pipeline
    should include an imputer, so any unseen/missing columns become NaN and are handled.
    """
    feats = {}

    if y is None or len(y) == 0:
        return feats

    # Basic stats
    feats["duration_s"] = float(len(y) / sr)
    feats["rms_mean"] = float(np.mean(librosa.feature.rms(y=y)))
    feats["rms_std"] = float(np.std(librosa.feature.rms(y=y)))
    feats["zcr_mean"] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    feats["zcr_std"] = float(np.std(librosa.feature.zero_crossing_rate(y)))

    # Spectral
    S, phase = librosa.magphase(librosa.stft(y=y, n_fft=2048, hop_length=512))
    spec_centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    spec_roll = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)

    feats["spec_centroid_mean"] = float(np.mean(spec_centroid))
    feats["spec_centroid_std"] = float(np.std(spec_centroid))
    feats["spec_bw_mean"] = float(np.mean(spec_bw))
    feats["spec_bw_std"] = float(np.std(spec_bw))
    feats["spec_rolloff_mean"] = float(np.mean(spec_roll))
    feats["spec_rolloff_std"] = float(np.std(spec_roll))

    # Tempo
    try:
        tempo = librosa.beat.tempo(y=y, sr=sr, hop_length=512, aggregate=None)
        feats["tempo_mean"] = float(np.mean(tempo)) if tempo.size else np.nan
        feats["tempo_std"] = float(np.std(tempo)) if tempo.size else np.nan
    except Exception:
        feats["tempo_mean"] = np.nan
        feats["tempo_std"] = np.nan

    # Chroma
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    feats["chroma_mean"] = float(np.mean(chroma))
    feats["chroma_std"] = float(np.std(chroma))
    for i in range(min(12, chroma.shape[0])):
        feats[f"chroma_{i+1:02d}_mean"] = float(np.mean(chroma[i]))
        feats[f"chroma_{i+1:02d}_std"] = float(np.std(chroma[i]))

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(mfcc.shape[0]):
        feats[f"mfcc_{i+1:02d}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1:02d}_std"] = float(np.std(mfcc[i]))

    # Spectral contrast
    try:
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        for i in range(contrast.shape[0]):
            feats[f"contrast_{i+1:02d}_mean"] = float(np.mean(contrast[i]))
            feats[f"contrast_{i+1:02d}_std"] = float(np.std(contrast[i]))
    except Exception:
        pass

    # Tonnetz (optional; uses chroma_cqt)
    try:
        y_harm = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        for i in range(tonnetz.shape[0]):
            feats[f"tonnetz_{i+1:02d}_mean"] = float(np.mean(tonnetz[i]))
            feats[f"tonnetz_{i+1:02d}_std"] = float(np.std(tonnetz[i]))
    except Exception:
        pass

    return feats

def load_audio_any(uploaded_bytes: bytes, target_sr: int, mono: bool = True, max_duration_s: float = 120.0) -> Tuple[np.ndarray, int]:
    """
    Loads audio from bytes using librosa/audioread/soundfile. Trims/slices to max_duration_s.
    """
    with io.BytesIO(uploaded_bytes) as bio:
        y, sr = librosa.load(bio, sr=target_sr, mono=mono, duration=max_duration_s)
    # Ensure finite numeric values
    if y is None:
        y = np.zeros(int(target_sr * 1.0), dtype=np.float32)
        sr = target_sr
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y, sr

def features_from_audio_bytes(uploaded_bytes: bytes, target_sr: int) -> Dict[str, float]:
    y, sr = load_audio_any(uploaded_bytes, target_sr=target_sr, mono=True)
    return extract_features_array(y, sr)

# --------------------
# Prediction helpers
# --------------------
def predict_dataframe(model, encoder, X: pd.DataFrame, top_k: int = 5):
    X = align_columns_to_model(X, model)
    y_pred = model.predict(X)
    # Map numeric codes -> labels
    if np.issubdtype(np.array(y_pred).dtype, np.number):
        labels = encoder.inverse_transform(y_pred.astype(int))
    else:
        labels = y_pred.astype(str)

    out = pd.DataFrame({"pred_idx": y_pred, "pred_label": labels})

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes, names = get_display_names(model, encoder)
        order = np.argsort(proba, axis=1)[:, ::-1][:, :min(top_k, proba.shape[1])]
        # show top-k labels + probs
        out["top_labels"] = [[names[i] for i in row] for row in order]
        out["top_probs"] = [[float(proba[r, i]) for i in row] for r, row in enumerate(order)]

    return out

def build_zip_by_genre(rows: List[Tuple[str, str, bytes]]) -> bytes:
    """
    rows: list of (genre_label, original_filename, file_bytes)
    returns: zip bytes (in-memory)
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for genre, fname, fbytes in rows:
            safe_genre = sanitize_filename(genre or "Unknown")
            safe_name = sanitize_filename(fname or "audio")
            arcname = f"{safe_genre}/{safe_name}"
            zf.writestr(arcname, fbytes)
    buf.seek(0)
    return buf.read()

# --------------------
# UI
# --------------------
st.title("🎛️ MilkCrate — Drop audio files → get genre folders (ZIP)")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    encoder_path = st.text_input("Encoder path", value=DEFAULT_ENCODER_PATH)
    target_sr = st.number_input("Target sample rate (Hz)", min_value=8000, max_value=48000, value=TARGET_SR_DEFAULT, step=1000)
    top_k = st.number_input("Top-K probabilities", min_value=1, max_value=10, value=5, step=1)
    max_duration = st.number_input("Analyze up to (seconds)", min_value=10, max_value=600, value=120, step=10)

model = load_model(model_path)
encoder = load_encoder(encoder_path)

with st.expander("🔎 Debug: label map (model ↔ encoder)"):
    classes, names = get_display_names(model, encoder)
    if classes is not None:
        st.dataframe(pd.DataFrame({"class_code": classes, "label": names}), use_container_width=True, hide_index=True)
    else:
        st.info("Model has no classes_ information.")

st.markdown("---")

st.subheader("Upload audio files (any common format)")
uploaded = st.file_uploader(
    "Drop multiple audio files here",
    type=["wav", "mp3", "flac", "ogg", "m4a", "aac", "wma", "aiff", "aif", "aifc"],
    accept_multiple_files=True
)

if uploaded:
    # Extract features for each file
    rows = []
    meta = []   # (original_name, feature_row_dict, raw_bytes)
    progress = st.progress(0)
    for i, f in enumerate(uploaded, start=1):
        try:
            raw = f.read()
            feats = features_from_audio_bytes(raw, target_sr=target_sr)
            feats["file_name"] = sanitize_filename(f.name)
            meta.append((f.name, feats, raw))
        except Exception as e:
            st.error(f"Failed to process {f.name}: {e}")
        progress.progress(i / len(uploaded))
    progress.empty()

    if not meta:
        st.error("No valid audio decoded.")
        st.stop()

    # Build feature dataframe
    feat_df = pd.DataFrame([row for (_, row, _) in meta]).fillna(np.nan)
    file_names = feat_df.pop("file_name").tolist() if "file_name" in feat_df.columns else [f"file_{i}" for i in range(len(meta))]

    # Predict
    try:
        preds = predict_dataframe(model, encoder, feat_df, top_k=top_k)
        preds.insert(0, "file_name", file_names)
        st.markdown("**Predictions**")
        st.dataframe(preds, use_container_width=True)
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
        st.stop()

    # Build ZIP grouped by predicted genre
    try:
        package_rows = []
        for (orig_name, _, raw), (_, row) in zip(meta, preds.iterrows()):
            label = row["pred_label"]
            package_rows.append((str(label), orig_name, raw))

        zip_bytes = build_zip_by_genre(package_rows)
        st.download_button(
            "⬇️ Download organized ZIP",
            data=zip_bytes,
            file_name="milkcrate_genres.zip",
            mime="application/zip",
            use_container_width=True
        )
        st.success("ZIP ready: files have been placed into folders named by predicted genre.")
    except Exception as e:
        st.error(f"Failed to build ZIP: {e}")

else:
    st.info("Upload multiple audio files to classify and export.")
