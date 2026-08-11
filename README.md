# MilkCrate DJ 🥛📦

**AI-powered music library organizer for DJs.** Scan your library or USB, get every track classified by genre and BPM, and export ready-to-use Rekordbox playlists — powered by a machine learning model trained on your own taxonomy that gets smarter with every correction.

🌐 [milkcrate.vip](https://milkcrate.vip)

---

## What It Does

Drop in a folder of untagged tracks — deemix rips, Bandcamp buys, USB from a b2b partner — and MilkCrate:

1. **Scans** every audio file (FLAC, AIFF, WAV, MP3, M4A, OGG)
2. **Classifies** each track into DJ-relevant subgenres (Tech House, Minimal/Deep, Afro House, Bass House, UKG, Drum & Bass, Techno, Trance, Dubstep, and more)
3. **Detects BPM** with octave-error correction (no more 60 BPM house tracks)
4. **Groups** tracks into reliable "families" (House / Bass / Techno / Trance) for crate organization
5. **Finds duplicates** — exact and near-duplicate detection
6. **Exports Rekordbox XML** — import directly into Rekordbox with playlists pre-organized

## Key Features

| Feature | Description |
|---|---|
| 🎯 **82.7% subgenre accuracy** | Top-3 accuracy of 97.7% — the right answer is almost always in the top 3 |
| 🏠 **Family-level reliability** | House/Bass/Techno/Trance groupings are highly reliable for crate building |
| ⚡ **Local & private** | All analysis runs on your machine. Your audio never leaves your computer |
| 🔁 **Feedback loop** | Correct any classification via dropdown — corrections train the next model version |
| 📊 **Confidence flagging** | Low-confidence tracks get flagged as "Unsorted" instead of guessing wrong |
| 💾 **Smart caching** | Tracks are analyzed once; re-scans are instant |
| 🎧 **In-app preview** | Play any track directly from the results table |
| 📦 **Rekordbox export** | One-click XML export with Family-only, Family>Subgenre, or Subgenre-only playlist structures |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MilkCrate DJ (macOS App)                 │
│                                                              │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ PyQt6 UI │──▶│   Scanner    │──▶│ Feature Extraction │  │
│  │          │   │ (worker      │   │ (librosa, 939      │  │
│  │ Results  │   │  threads)    │   │  features/track)   │  │
│  │ Summary  │   └──────────────┘   └─────────┬──────────┘  │
│  │ Dupes    │                                 │             │
│  └────┬─────┘   ┌──────────────┐   ┌─────────▼──────────┐  │
│       │         │ SQLite Cache │◀──│  LightGBM Model    │  │
│       │         │ (~/.milkcrate)│   │  + StandardScaler  │  │
│       ▼         └──────────────┘   │  + Label Encoder   │  │
│  ┌──────────┐                      └────────────────────┘  │
│  │ Rekordbox│   ┌──────────────┐                            │
│  │ XML      │   │ Feedback Log │◀── Override dropdown       │
│  │ Export   │   │ (jsonl)      │                            │
│  └──────────┘   └──────┬───────┘                            │
└─────────────────────────┼───────────────────────────────────┘
                          │ (feature vectors only — never audio)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend (api.milkcrate.vip, Hetzner)            │
│                                                              │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ FastAPI  │──▶│  PostgreSQL  │──▶│  Weekly Retraining │  │
│  │ REST API │   │  (feedback   │   │  (cron, LightGBM)  │  │
│  └────┬─────┘   │   store)     │   └─────────┬──────────┘  │
│       │         └──────────────┘             │             │
│       │                                       ▼             │
│       │         ┌────────────────────────────────────────┐ │
│       └────────▶│  Versioned Model Bundles (served back  │ │
│                 │  to apps via auto-update)               │ │
│                 └────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### The ML Pipeline

**Feature extraction (939 features per track):**
- 40 MFCCs + 40 delta MFCCs (×5 statistics each) — timbral fingerprint
- 128-band mel spectrogram statistics — full spectral picture
- 12-bin chroma — harmonic/key content
- 7-band spectral contrast — texture separation
- 6-dim tonnetz — tonal centroid features
- Spectral centroid/bandwidth/rolloff, ZCR, RMS energy
- BPM + beat regularity (octave-corrected)
- Harmonic/percussive energy ratio

**Model:** LightGBM gradient-boosted trees with balanced class weights, feature scaling, and early stopping. Trained on a hand-labeled dataset with 4x data augmentation (pitch shift ±1 semitone, 5% time stretch).

**Two-stage output:**
- **Subgenre (hint)** — best guess, treat as a starting point
- **Family (reliable)** — coarse grouping with much higher confidence, recommended for crates

### The Feedback Loop (Decentralized Improvement)

1. DJ overrides a wrong classification via the dropdown
2. Correction (feature vector + label — **never audio**) queues locally
3. On next app launch, corrections sync to the backend
4. Weekly cron retrains the model when ≥20 new corrections accumulate
5. All apps auto-download the improved model on launch

Every correction from every user makes the model better for everyone.

---

## Project Structure

```
milkcrate-dj/
├── app/
│   ├── main.py               # PyQt6 UI, app entry point
│   ├── scanner.py            # Folder walking, per-track orchestration
│   ├── feature_extract.py    # 939-feature librosa extraction
│   ├── classifier.py         # Model loading + prediction (2-stage)
│   ├── coarse_classifier.py  # Electronic vs non-electronic gate
│   ├── worker.py             # Background scan threads (Qt signals)
│   ├── cache_db.py           # SQLite feature/result cache
│   ├── duplicate_detect.py   # Exact + near-duplicate detection
│   ├── rekordbox_export.py   # Rekordbox XML playlist generation
│   ├── tag_reader.py         # ID3/Vorbis/MP4 tag reading (mutagen)
│   ├── feedback_store.py     # Correction logging (jsonl)
│   └── sync_client.py        # Backend sync + model auto-update
├── model/
│   ├── artifacts/            # Trained model, encoder, scaler, config
│   ├── train_enhanced.py     # Main training pipeline (939 features)
│   ├── augment_dataset.py    # Pitch/stretch data augmentation
│   └── retrain_from_feedback.py
├── backend/                  # Deployed to api.milkcrate.vip
│   ├── main.py               # FastAPI (feedback, model serving, stats)
│   ├── retrain_server.py     # Weekly server-side retraining
│   └── DEPLOY.md             # Hetzner deployment guide
├── setup.py                  # py2app macOS bundling
└── requirements.txt
```

## Tech Stack

| Layer | Technology |
|---|---|
| Desktop UI | Python 3.10, PyQt6 |
| Audio analysis | librosa, soundfile, audioread |
| ML | LightGBM, scikit-learn |
| Tags | mutagen |
| Local cache | SQLite |
| Packaging | py2app (macOS) |
| Backend API | FastAPI + uvicorn |
| Database | PostgreSQL |
| Hosting | Hetzner CX22 (€4.55/mo) |
| Model serving | Versioned zip bundles over HTTPS |

## Requirements

- **End users:** macOS (Apple Silicon), nothing else — fully self-contained .app
- **Development:** Python 3.10+, `pip install -r requirements.txt`

## Development

```bash
# Run in dev mode
cd app && python3 main.py

# Retrain the model on your labeled dataset
cd model && python3 train_enhanced.py \
  --data-dir ~/Desktop/classified_music_dataset \
  --out-dir artifacts

# Augment training data (3x per track: pitch ±1, stretch 1.05x)
python3 augment_dataset.py --data-dir ~/Desktop/classified_music_dataset

# Build the macOS app
python3 setup.py py2app
```

## Roadmap

- [x] Core scan → classify → export pipeline
- [x] Custom genre taxonomy + user dataset training
- [x] Data augmentation (82.7% accuracy)
- [x] Duplicate detection
- [x] Feedback/override system
- [x] Backend for decentralized model improvement
- [ ] Apple Developer signing + notarization
- [ ] TestFlight beta distribution
- [ ] Auto-update integration in app
- [ ] Windows support
- [ ] Coarse gate (electronic vs non-electronic filtering)
- [ ] MERT/AST deep learning features (at 150+ tracks per class)

---

*Built by a DJ, for DJs. Your crates, organized.*
