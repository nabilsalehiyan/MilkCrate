"""
App-side sync client for MilkCrate DJ.
Add to the app/ folder. Called on app launch and after corrections.

- Uploads pending feedback corrections (feature vectors + labels only, no audio)
- Checks for model updates and downloads them
"""

import os
import json
import zipfile
import tempfile
import urllib.request
import urllib.error

BACKEND_URL = os.environ.get("MILKCRATE_BACKEND", "https://api.yourdomain.com")
API_KEY = os.environ.get("MILKCRATE_API_KEY", "beta-key-1")

FEEDBACK_PATH = os.path.expanduser("~/.milkcrate/feedback.jsonl")
SYNCED_MARKER = os.path.expanduser("~/.milkcrate/feedback_synced_count.txt")
MODEL_VERSION_FILE = os.path.expanduser("~/.milkcrate/model_version.txt")


def _request(method, path, data=None, timeout=15):
    url = f"{BACKEND_URL}{path}"
    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def upload_pending_feedback(features_lookup_fn):
    """
    Upload corrections that haven't been synced yet.
    features_lookup_fn(file_path) -> list of 939 floats, or None
    """
    if not os.path.exists(FEEDBACK_PATH):
        return 0

    synced = 0
    if os.path.exists(SYNCED_MARKER):
        with open(SYNCED_MARKER) as f:
            synced = int(f.read().strip() or 0)

    with open(FEEDBACK_PATH) as f:
        lines = f.readlines()

    pending = lines[synced:]
    if not pending:
        return 0

    corrections = []
    for line in pending:
        try:
            rec = json.loads(line)
            feats = features_lookup_fn(rec.get("file"))
            if feats is None:
                continue
            corrections.append({
                "features": feats,
                "predicted_label": rec.get("predicted", "unknown"),
                "corrected_label": rec.get("corrected", rec.get("label", "unknown")),
                "confidence": rec.get("confidence"),
                "app_version": "1.0.0",
                "model_version": _current_model_version(),
            })
        except Exception:
            continue

    if not corrections:
        return 0

    try:
        result = _request("POST", "/api/v1/feedback", {"corrections": corrections})
        with open(SYNCED_MARKER, "w") as f:
            f.write(str(len(lines)))
        return result.get("received", 0)
    except (urllib.error.URLError, OSError):
        return 0  # offline — try next launch


def _current_model_version():
    if os.path.exists(MODEL_VERSION_FILE):
        with open(MODEL_VERSION_FILE) as f:
            return f.read().strip()
    return "local"


def check_and_update_model(artifacts_dir):
    """
    Check backend for newer model. If found, download and replace artifacts.
    Returns new version string or None.
    """
    try:
        info = _request("GET", "/api/v1/model/latest")
    except (urllib.error.URLError, OSError):
        return None

    latest = info["version"]
    if latest == _current_model_version():
        return None

    # Download and install
    try:
        url = f"{BACKEND_URL}/api/v1/model/download/{latest}"
        req = urllib.request.Request(url, headers={"X-API-Key": API_KEY})
        with urllib.request.urlopen(req, timeout=120) as resp:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                tmp.write(resp.read())
                tmp_path = tmp.name

        with zipfile.ZipFile(tmp_path) as z:
            z.extractall(artifacts_dir)
        os.unlink(tmp_path)

        with open(MODEL_VERSION_FILE, "w") as f:
            f.write(latest)
        return latest
    except Exception:
        return None
