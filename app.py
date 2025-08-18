# app.py
# Streamlit app for MilkCrate — predicts genres and displays human-readable labels.
# Works with a sklearn Pipeline saved as models/model_version3beatport.joblib
# and a LabelEncoder saved as artifacts/label_encoder.joblib.
#
# If the encoder is missing, we try to build it from data/beatport_features.csv (column: 'genre').

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_MODEL_PATH = "models/model_version3beatport.joblib"
DEFAULT_ENCODER_PATH = "artifacts/label_encoder.joblib"
DEFAULT_CSV_PATH = "data/beatport_features.csv"
TARGET_COL_DEFAULT = "genre"

st.set_page_config(page_title="MilkCrate Genre Classifier", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: str):
    if not os.path.exists(path):
        st.error(f"Model not found at {path}")
        st.stop()
    model = joblib.load(path)
    return model

@st.cache_resource(show_spinner=False)
def load_or_build_encoder(enc_path: str, csv_path: str, target_col: str):
    """Load LabelEncoder if available; otherwise build it from CSV with string labels."""
    try:
        if os.path.exists(enc_path):
            enc = joblib.load(enc_path)
            return enc, "loaded"
    except Exception as e:
        st.warning(f"Couldn't load encoder at {enc_path}: {e}")

    # Try to build from CSV
    if os.path.exists(csv_path):
        try:
            from sklearn.preprocessing import LabelEncoder
            df = pd.read_csv(csv_path)
            if target_col not in df.columns:
                st.warning(f"CSV found at {csv_path}, but '{target_col}' column is missing. Encoder cannot be built.")
                return None, "missing"
            labels = df[target_col].astype(str)
            enc = LabelEncoder().fit(labels)
            os.makedirs(os.path.dirname(enc_path), exist_ok=True)
            joblib.dump(enc, enc_path)
            return enc, "built"
        except Exception as e:
            st.warning(f"Failed to build encoder from {csv_path}: {e}")
            return None, "missing"

    return None, "missing"

def align_columns_to_model(X: pd.DataFrame, model):
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        missing = [c for c in names if c not in X.columns]
        if missing:
            st.warning(f"{len(missing)} expected columns are missing in your input. Showing first few: {missing[:10]}")
        X = X.reindex(columns=names)
    return X

def get_display_names(model, encoder):
    """Return array of display labels aligned to model.classes_."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None, None  # No class info
    classes = np.array(classes)

    # If classes are numeric, map them with encoder when available
    if np.issubdtype(classes.dtype, np.number):
        if encoder is not None:
            try:
                names = encoder.inverse_transform(classes.astype(int))
                return classes, names
            except Exception as e:
                st.warning(f"Couldn't inverse_transform classes with encoder: {e}")
                return classes, classes.astype(str)
        else:
            return classes, classes.astype(str)
    else:
        # Model already has string labels
        return classes, classes

def predict_one(model, features_df: pd.DataFrame, encoder, top_k=5):
    X = features_df.select_dtypes(include=[np.number])
    X = align_columns_to_model(X, model)
    y = model.predict(X)
    # Map y (possibly numeric) to display string
    if encoder is not None and np.issubdtype(np.array(y).dtype, np.number):
        try:
            y_labels = encoder.inverse_transform(y.astype(int))
        except Exception:
            y_labels = y.astype(str)
    else:
        y_labels = y

    result = {"pred_idx": y, "pred_label": y_labels}

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
        classes, names = get_display_names(model, encoder)
        if classes is not None and names is not None:
            order = np.argsort(prob)[::-1]
            top = order[: min(top_k, len(order))]
            top_table = pd.DataFrame(
                {
                    "rank": np.arange(1, len(top) + 1),
                    "label": [names[i] for i in top],
                    "class_code": [int(classes[i]) if np.issubdtype(classes.dtype, np.integer) else classes[i] for i in top],
                    "probability": [float(prob[i]) for i in top],
                }
            )
            result["topk"] = top_table
    return result

def read_uploaded_json_or_row(uploaded_bytes: bytes):
    """Accept a JSON dict of features OR a CSV with a single row."""
    text = uploaded_bytes.decode("utf-8", errors="ignore")
    # Try JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            df = pd.DataFrame([obj])
            return df
        elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return pd.DataFrame(obj[:1])
    except Exception:
        pass
    # Try CSV fallback
    try:
        df = pd.read_csv(io.StringIO(text))
        if len(df) > 1:
            st.info(f"Detected CSV with {len(df)} rows. For single-item mode we'll use the first row.")
        return df.head(1)
    except Exception as e:
        st.error("Couldn't parse the uploaded file as JSON or CSV (single row).")
        st.caption(str(e))
        return None

# ----------------------------
# UI
# ----------------------------
st.title("🎛️ MilkCrate — Genre Classifier")

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("Model path", value=DEFAULT_MODEL_PATH)
    encoder_path = st.text_input("Encoder path", value=DEFAULT_ENCODER_PATH)
    csv_path = st.text_input("Fallback CSV (to build encoder if missing)", value=DEFAULT_CSV_PATH)
    target_col = st.text_input("Target column name", value=TARGET_COL_DEFAULT)
    top_k = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)

model = load_model(model_path)
encoder, enc_state = load_or_build_encoder(encoder_path, csv_path, target_col)

if enc_state == "built":
    st.success(f"Built encoder from {csv_path} → {encoder_path}")
elif enc_state == "loaded":
    st.caption(f"Loaded encoder from {encoder_path}")
else:
    st.warning("No encoder available. Numeric predictions will be shown as class codes.")

# Debug panel
with st.expander("🔎 Debug: label map"):
    classes, names = get_display_names(model, encoder)
    if classes is not None:
        df_map = pd.DataFrame({"class_code": classes, "label": names})
        st.dataframe(df_map, use_container_width=True, hide_index=True)
    else:
        st.info("Model has no .classes_ attribute.")

st.markdown("---")

tabs = st.tabs(["Single item", "Batch CSV"])

# ----------------------------
# Single item tab
# ----------------------------
with tabs[0]:
    st.subheader("Single item prediction")
    st.caption("Upload a JSON dict (feature_name → value) or a CSV with one row of features.")

    up = st.file_uploader("Upload features (JSON or 1-row CSV)", type=["json", "csv"], key="single")
    if up is not None:
        df = read_uploaded_json_or_row(up.getvalue())
        if df is not None:
            st.write("Parsed input (first row shown):")
            st.dataframe(df.head(1), use_container_width=True)
            try:
                res = predict_one(model, df, encoder, top_k=top_k)
                plabel = res["pred_label"][0]
                st.success(f"Predicted genre: **{plabel}**")
                if "topk" in res:
                    st.markdown("**Top probabilities**")
                    st.dataframe(res["topk"], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error("Prediction failed.")
                st.exception(e)

# ----------------------------
# Batch tab
# ----------------------------
with tabs[1]:
    st.subheader("Batch predictions from CSV")
    st.caption(
        "Upload a CSV of feature rows. If it contains your target column "
        f"('{target_col}'), we'll drop it before prediction and show class balance."
    )
    up_csv = st.file_uploader("Upload features CSV", type=["csv"], key="batch")
    if up_csv is not None:
        try:
            df = pd.read_csv(up_csv)
        except Exception as e:
            st.error("Couldn't read CSV.")
            st.caption(str(e))
            df = None

        if df is not None:
            st.write("Preview:")
            st.dataframe(df.head(10), use_container_width=True)

            # Show class balance if target present
            if target_col in df.columns:
                st.markdown("**Target distribution (top 25)**")
                st.dataframe(df[target_col].value_counts().head(25).to_frame("count"))

            # Prepare X
            X = df.copy()
            if target_col in X.columns:
                X = X.drop(columns=[target_col])

            # Keep numeric only (preprocessing pipeline usually handles this; this is conservative)
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if not num_cols:
                st.error("No numeric columns detected in CSV.")
            else:
                X = X[num_cols]
                X = align_columns_to_model(X, model)

                # Predict
                try:
                    y_pred = model.predict(X)
                    if encoder is not None and np.issubdtype(np.array(y_pred).dtype, np.number):
                        y_label = encoder.inverse_transform(y_pred.astype(int))
                    else:
                        y_label = y_pred.astype(str)

                    out = pd.DataFrame({"pred_idx": y_pred, "pred_label": y_label})
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X)
                        classes, names = get_display_names(model, encoder)
                        if classes is not None and names is not None:
                            top_idx = np.argsort(proba, axis=1)[:, ::-1][:, :top_k]
                            top_labels = []
                            top_probs = []
                            for r, row in enumerate(top_idx):
                                top_labels.append([names[i] for i in row])
                                top_probs.append([float(proba[r, i]) for i in row])
                            out["top_labels"] = top_labels
                            out["top_probs"] = top_probs

                    st.markdown("**Predictions**")
                    st.dataframe(out.head(100), use_container_width=True)

                    # Degenerate check
                    uniq = pd.Series(out["pred_label"]).nunique(dropna=False)
                    if uniq == 1:
                        only = out["pred_label"].iloc[0]
                        st.warning(
                            f"All predictions in this preview are the same class → **{only}**.\n\n"
                            "- Check class imbalance in training\n"
                            "- Ensure the same preprocessing is used at train & inference\n"
                            "- Verify encoder ↔ model mapping is correct\n"
                            "- Confirm you’re loading the latest model artifact"
                        )
                except Exception as e:
                    st.error("Prediction failed.")
                    st.exception(e)

st.markdown("---")
st.caption("MilkCrate • model: VotingClassifier • labels via LabelEncoder")
