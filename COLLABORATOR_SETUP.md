# MilkCrate DJ — Collaborator Setup

Welcome! This is the full MilkCrate DJ codebase. Read `README.md` first for the architecture overview. This file gets you from zero to a running app.

## What's in this package

```
app/        # The macOS desktop app (PyQt6) — run this
model/      # Training pipeline (feature extraction, augmentation, retraining)
backend/    # FastAPI server for the decentralized feedback loop (not yet deployed)
setup.py    # py2app config for building the standalone .app
```

## ⚠️ One thing you need that's NOT in here: the trained model

The trained model artifacts (~100MB) aren't in this zip. Get these **5 files from Nabil** and drop them into `model/artifacts/`:

```
genre_model.joblib      # LightGBM model (82.7% accuracy)
label_encoder.joblib
scaler.joblib           # REQUIRED — predictions are garbage without it
feature_columns.json    # must be the 939-feature version
family_map.json         # (already included as reference)
```

Alternatively, if you have a labeled dataset (folders of audio, one folder per genre), you can train your own — see "Retraining" below.

## Setup (macOS, Apple Silicon)

```bash
# 1. Python 3.10 recommended (pyenv is the sane way)
python3 --version

# 2. Install dependencies
pip3 install -r requirements.txt --break-system-packages

# 3. Drop the model artifacts into model/artifacts/ (see above)

# 4. Run the app
cd app
python3 main.py
```

Select a music folder → Scan Library. You should see varied genres, BPMs, and the Tag Genre column comparing embedded metadata against model predictions.

**Sanity check:** if every track gets the *same* prediction, the scaler is missing or feature_columns.json is the wrong version.

## Key files to understand first

| File | What it does |
|---|---|
| `app/main.py` | UI, table population, override dropdown, export button |
| `app/scanner.py` | Per-track pipeline: cache check → extract → predict → tag compare |
| `app/feature_extract.py` | 939 librosa features (see `_extract_enhanced`) |
| `app/classifier.py` | Loads model + scaler, 2-stage predict (subgenre → family) |
| `model/train_enhanced.py` | Full training pipeline |
| `model/augment_dataset.py` | 3x data augmentation (pitch ±1 semitone, 1.05x stretch) |

## Retraining

```bash
# Dataset layout: one folder per genre label
# ~/dataset/Tech_House/*.flac, ~/dataset/Techno/*.aiff, etc.

cd model
python3 augment_dataset.py --data-dir ~/dataset          # one-time, 3x's the data
python3 train_enhanced.py \
  --data-dir ~/dataset \
  --cache artifacts/enhanced_features.parquet \
  --out-dir artifacts
```

Feature extraction is the slow part (hours for ~2000 files) but caches to parquet — subsequent trainings take minutes. Delete the parquet to force fresh extraction after changing the dataset.

## Building the standalone .app

```bash
python3 -c "
import sys
sys.setrecursionlimit(50000)
sys.argv = ['setup.py', 'py2app']
exec(compile(open('setup.py').read(), 'setup.py', 'exec'))
"

# Then copy artifacts + native libs into the bundle:
mkdir -p "dist/MilkCrate DJ.app/Contents/Resources/model/artifacts"
cp model/artifacts/*.joblib model/artifacts/*.json "dist/MilkCrate DJ.app/Contents/Resources/model/artifacts/"
cp <path-to>/libomp.dylib "dist/MilkCrate DJ.app/Contents/Frameworks/"        # from: find /opt/homebrew -name libomp.dylib
cp <path-to>/libsndfile_arm64.dylib "dist/MilkCrate DJ.app/Contents/Frameworks/libsndfile.dylib"  # from site-packages/_soundfile_data/
xattr -cr "dist/MilkCrate DJ.app"
codesign --deep --force --sign - "dist/MilkCrate DJ.app"
```

Known gotchas (all hit before, all solved this way):
- **py2app RecursionError** → the `setrecursionlimit` wrapper above fixes it
- **All tracks ERROR after packaging** → a native dylib (libsndfile/libomp) isn't bundled; run the binary from Terminal to see the real error: `"dist/MilkCrate DJ.app/Contents/MacOS/MilkCrate DJ"`
- **All tracks get the same prediction** → scaler.joblib missing from the bundle
- **"App is damaged" on another Mac** → recipient runs `sudo xattr -rd com.apple.quarantine <path-to-app>` (proper fix = Apple Developer signing, on the roadmap)

## Backend (not yet deployed)

`backend/` contains a complete FastAPI + PostgreSQL service for collecting user corrections and serving retrained models. `backend/DEPLOY.md` has the full Hetzner deployment guide. Status: code done, server not yet provisioned. Domain will be `api.milkcrate.vip`.

## Suggested workflow for us

**Set up a GitHub repo** (private) and push this codebase — passing zips back and forth will break down fast with two people. Add `model/artifacts/` and `*.parquet` to `.gitignore` (too big for git; share via Drive or git-lfs).

## Current state / open work

Done: scan→classify→export pipeline, 82.7% model, dedup, feedback logging, tag-genre comparison column, backend code.

Open (roughly in priority order):
1. Deploy backend to Hetzner + wire `app/sync_client.py` into app startup
2. Apple Developer account → sign + notarize → frictionless distribution
3. Tag-as-tiebreaker: use metadata genre to break low-confidence predictions
4. Grow dataset toward 150+/class, then evaluate MERT/AST deep features
5. Windows port
