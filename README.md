# MilkCrate

Sorts your EDM library into DJ-ready crates, and writes them straight to a Pioneer USB.

Closed **beta** for invited testers. This repository is a **download page only** — there is no application source here, no git history of the product, and no license to copy the app. All rights reserved.

**[Download the latest Mac build](https://github.com/nabilsalehiyan/MilkCrate/releases/tag/v0.1.0)**

Apple Silicon (M1 or newer, including M5), macOS 13+. Intel Macs and Windows are not in this beta.

---

## Install

1. Download `MilkCrate-macOS-arm64.zip` from the release above, then unzip it.
2. Drag **MilkCrate** into your Applications folder.
3. Open Terminal and paste these two lines:

```
xattr -cr /Applications/MilkCrate.app
open /Applications/MilkCrate.app
```

Prefer not to use Terminal? Double-click **If it wont open.command** (right-click → Open).

This beta is not notarized by Apple yet, so macOS blocks it the first time. The command above clears that flag, and you only do it once. On macOS 13 and 14 you can right-click → Open instead; on macOS 15 and newer, let macOS block it, then go to System Settings → Privacy & Security → **Open Anyway**.

First launch asks for your name and email. That's required for the beta — it's how you get update notices. Your audio never leaves your Mac.

## How it works

MilkCrate listens to each track, measures about 900 things about the sound — rhythm, texture, energy, frequency balance — and matches that against a model trained on a hand-sorted EDM library. It sorts into five families: **House, Indie Dance, Minimal, Tech House, Techno**.

If the model is less than 35% sure, the track goes to **Unsorted** rather than getting a bad guess. Unsorted is a good sign, not a bug.

## The two ways to use it

**Sort a folder of new music:** drop files in → fix anything wrong → **Export** → **Send back to USB**.

**Sort a USB that Rekordbox already exported:** **Open USB** → fix anything wrong → **Write playlists to USB**. This never copies or moves your audio. It reads Rekordbox's own database, sorts the tracks where they already sit, and adds playlists, so your beatgrids, waveforms, and existing cues stay intact.

## The correction loop — the important part

Every track has a **Genre** dropdown. If MilkCrate gets one wrong, pick the right genre. The moment you do, the correction is saved, the track joins your personal training set, and **the model retrains itself in the background**. The status bar confirms it: "Model updated. New drops will use what you taught it."

The app genuinely gets smarter the more you correct it, and it learns *your* ears — your idea of where Tech House ends and Techno begins. Correct 20 tracks and you'll see the difference on the next folder you drop.

Fix the **red** confidence numbers first: red means the model was unsure, so your answer teaches it the most. Corrections only change the model, never your audio files.

Ticking **Share corrections** also sends your fixes to the shared model everyone gets. It transmits only the numeric fingerprint and the genre you picked — never the song, never the filename. Off by default, entirely your call.

## What every button does

| Button | What it does |
| --- | --- |
| **Share corrections** | Opt in to improve the shared community model. Sends numbers and a genre label only. Off by default. |
| **Report a problem** | Short form to send a rating and a note to the developer. Use any time. |
| **Hot cues** | Suggests cue points on every track: **A** Intro, **B** Drop, **C** Break, **D** Drop 2, **E** Outro. Written for Pioneer decks on Send back to USB. |
| **Open USB** | Reads a Rekordbox USB's database and sorts every track already on the stick, in place. |
| **Learn from USB** | Different: reads the genre tags *you* set in Rekordbox and trains on them. Best on a well-tagged stick. Takes several minutes; files are linked, not copied. |
| **Update model** | Retrains now from all your corrections. Usually unnecessary — corrections retrain on their own. |
| **Export** | Converts the sorted crate to 24-bit / 44.1 kHz AIFF, in folders by genre, to `~/Desktop/MilkCrate Export`. Originals untouched. |
| **Send back to USB** | Copies those AIFF crates to your Pioneer USB and adds Playlists → MilkCrate, with your hot cues. |
| **Write playlists to USB** | After Open USB: adds Playlists → MilkCrate *(date)* to the stick, one per family, pointing at tracks already there. Nothing copied. |

In the track list: the **triangle** previews a track, **Family** is the crate it's headed for, **Confidence** is how sure the model is (red below 35%), and **Genre** is where you correct it. The drop zone accepts AIFF, WAV, MP3, FLAC, and M4A, including subfolders.

Either USB path backs up `export.pdb` before writing, and restores it automatically if the write fails.

## Updates

When a new build is out, MilkCrate tells you on launch and offers an **Update now** button that downloads, installs, and reopens the app for you. No Terminal, no re-dragging. Choose **Later** and it asks again next time.

## Where things live

```
~/Desktop/MilkCrate Export     exported AIFF crates
~/Desktop/data/training        tracks you corrected
~/.milkcrate                   settings, your updated model, logs
```

## If something goes wrong

This is a test build — it can mis-tag tracks and it can crash. That's what the beta is for, so please report it.

**App won't open:** re-run the two Terminal lines above, and check that Apple menu → About This Mac says Apple M-series, not Intel. Still stuck? Send the file `~/.milkcrate/launch.log`.

**A track won't play or classify:** usually a corrupt or DRM-protected file. Try it in another player first.

**Everything landed in Unsorted:** the model was unsure. Correct a handful by hand — it learns fast — or send a note with Report a problem.

**Sorting is slow:** normal. Every track gets a full ~900-feature analysis, so a big USB takes a while. It runs in the background.

## Privacy, in plain terms

**Never leaves your Mac:** your audio, your file names, your folders.

**Sent to the developer:** your name and email (for updates), crash reports, feedback you write, and — only if you tick Share corrections — anonymous numeric fingerprints plus the genre label you chose.

Invited testers only. Please don't repost the download.
