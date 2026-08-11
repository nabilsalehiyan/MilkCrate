"""
main.py
=======
MilkCrate DJ — desktop app entry point.

Workflow:
  1. Pick a folder (USB drive / music library).
  2. Scan in the background — extracts features, classifies genre/subgenre/
     family using a two-stage gate (Electronic/Dance vs. other broad
     genres, then fine-grained EDM subgenre), caches results, detects
     duplicates. Can be stopped mid-scan.
  3. Review results in a table (with subgenre/family override dropdowns).
     Tracks the gate couldn't confidently classify land in
     "Unsorted_LowConfidence" for manual review.
  4. Review flagged duplicates.
  5. Export a Rekordbox-compatible XML library (Preferences > Import Library
     in Rekordbox to bring it in).
"""

from __future__ import annotations

import os
import sys

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTableWidget,
    QTableWidgetItem, QComboBox, QTabWidget, QListWidget, QListWidgetItem,
    QMessageBox, QHeaderView,
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

from worker import ScanWorker
from rekordbox_export import build_rekordbox_xml
from classifier import TwoStageClassifier, UNSORTED_LABEL


# Resolve base path — works in dev mode, PyInstaller, and py2app bundles
if getattr(sys, "frozen", False):
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller
        _BASE = sys._MEIPASS
    else:
        # py2app — resources live in Contents/Resources/
        _BASE = os.path.normpath(
            os.path.join(os.path.dirname(sys.executable), "..", "Resources")
        )
else:
    # Normal dev mode
    _BASE = os.path.join(os.path.dirname(__file__), "..")

# Add paths so bundled modules are importable
for _p in [_BASE, os.path.join(_BASE, "app")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

ARTIFACTS_DIR = os.path.join(_BASE, "model", "artifacts")
CACHE_DB_PATH = os.path.join(os.path.expanduser("~"), ".milkcrate", "cache.db")


# Canonical subgenre list — one label per genre family
# Legacy BeatsDataset names are mapped to canonical ones at display time
_LEGACY_TO_CANONICAL = {
    "TechHouse": "Tech_House",
    "DrumAndBass": "Drum_and_bass",
    "HardcoreHardTechno": "Techno",
    "HipHop": "HipHop_RnB",
    "FunkRAndB": "HipHop_RnB",
    "ElectronicaDowntempo": "Downtempo",
    "IndieDanceNuDisco": "Indie_Dance",
    "ProgressiveHouse": "Progressive_House",
    "GlitchHop": "Breaks",
}

EDM_SUBGENRES = [
    # House family
    "Tech_House",
    "Minimal_deep",
    "Afro_house",
    "Classic_house",
    "Melodic_house",
    "DeepHouse",
    "Progressive_House",
    "House",
    "ElectroHouse",
    "FutureHouse",
    "BigRoom",
    "Indie_Dance",
    # Techno / Hard
    "Techno",
    "HardDance",
    # Bass family
    "Drum_and_bass",
    "UKG",
    "Dubstep",
    "Breaks",
    # Trance
    "Trance",
    "PsyTrance",
    # Other electronic
    "Dance",
    "Downtempo",
    # Non-electronic
    "HipHop_RnB",
    "Rock_Alternative",
    "Pop_Vocal",
    "Jazz_Acoustic_Classical",
    "Other",
    "Unsorted_LowConfidence",
]

# Extra override options for tracks the gate routed elsewhere
NON_EDM_OPTIONS = [UNSORTED_LABEL, "HipHop_RnB", "Rock_Alternative", "Pop_Vocal",
                   "Jazz_Acoustic_Classical", "Other"]

ALL_OVERRIDE_OPTIONS = EDM_SUBGENRES + NON_EDM_OPTIONS


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MilkCrate — DJ Library Organizer")
        self.resize(1050, 680)

        self.results = []
        self.duplicates = {"exact": [], "near": []}
        self.worker = None

        # Shared audio player for preview playback
        self.audio_output = QAudioOutput()
        self.media_player = QMediaPlayer()
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.currently_playing_row = None
        self.currently_playing_btn = None

        self.classifier = None
        try:
            self.classifier = TwoStageClassifier(ARTIFACTS_DIR)
            if self.classifier.coarse is None:
                self._coarse_missing_notice = True
            else:
                self._coarse_missing_notice = False
        except Exception as e:
            QMessageBox.warning(self, "Model load warning",
                                 f"Could not load trained model from {ARTIFACTS_DIR}:\n{e}\n"
                                 "Run model/train_model.py first.")
            self._coarse_missing_notice = False

        self._build_ui()

        if self._coarse_missing_notice:
            self.progress_label.setText(
                "Note: coarse genre gate not trained yet — using EDM model "
                "with confidence threshold only (non-electronic tracks may "
                "still be misclassified). See model/train_coarse_model.py."
            )

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # --- top bar: folder picker
        top_bar = QHBoxLayout()
        self.folder_label = QLabel("No folder selected")
        pick_btn = QPushButton("Select Music Folder / USB...")
        pick_btn.clicked.connect(self.pick_folder)
        self.scan_btn = QPushButton("Scan Library")
        self.scan_btn.setEnabled(False)
        self.scan_btn.clicked.connect(self.start_scan)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_scan)
        top_bar.addWidget(self.folder_label, 1)
        top_bar.addWidget(pick_btn)
        top_bar.addWidget(self.scan_btn)
        top_bar.addWidget(self.stop_btn)
        layout.addLayout(top_bar)

        # --- progress
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)

        # --- tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Results tab
        self.results_table = QTableWidget(0, 10)
        self.results_table.setHorizontalHeaderLabels(
            ["", "File", "Subgenre (hint)", "Tag Genre", "Family (reliable)", "BPM", "BPM Src", "Top-3 / Confidence", "Stage", "Override"]
        )
        self.results_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.results_table.setColumnWidth(0, 36)
        self.tabs.addTab(self.results_table, "Results")

        # Summary tab — family-level distribution overview
        self.summary_list = QListWidget()
        self.tabs.addTab(self.summary_list, "Summary")

        # Duplicates tab
        self.dup_list = QListWidget()
        self.tabs.addTab(self.dup_list, "Duplicates")

        # --- export bar
        export_bar = QHBoxLayout()
        export_bar.addWidget(QLabel("Group playlists by:"))
        self.grouping_combo = QComboBox()
        self.grouping_combo.addItems([
            "Family only (recommended)", "Family > Subgenre", "Subgenre only"
        ])
        export_bar.addWidget(self.grouping_combo)
        export_bar.addWidget(QLabel(
            "  Family = reliable (House/HardTechno/Trance/Electronic). "
            "Subgenre = best-guess, treat as a hint."
        ))
        export_bar.addStretch(1)
        self.export_btn = QPushButton("Export Rekordbox XML...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_xml)
        export_bar.addWidget(self.export_btn)
        layout.addLayout(export_bar)

        # --- feedback / learning bar
        feedback_bar = QHBoxLayout()
        self.feedback_label = QLabel("")
        feedback_bar.addWidget(self.feedback_label)
        feedback_bar.addStretch(1)
        layout.addLayout(feedback_bar)
        self.update_feedback_label()

        self.selected_folder = None

    def update_feedback_label(self):
        from feedback_store import feedback_count
        n = feedback_count()
        if n == 0:
            self.feedback_label.setText(
                "Corrections you make below are saved and can be used to "
                "improve the model later (model/retrain_from_feedback.py)."
            )
        else:
            self.feedback_label.setText(
                f"{n} correction(s) saved so far — run "
                f"'python model/retrain_from_feedback.py' to retrain using them "
                f"(plus the base dataset)."
            )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def toggle_play(self, path, row):
        """Play/stop preview for the track in this row. Clicking the
        currently-playing row's button stops it; clicking a different
        row stops the previous one and starts the new one."""
        if self.currently_playing_row == row:
            self._stop_playback()
            return

        if self.currently_playing_row is not None:
            self._stop_playback()

        self.media_player.setSource(QUrl.fromLocalFile(path))
        self.media_player.play()
        self.currently_playing_row = row
        btn = self.results_table.cellWidget(row, 0)
        if btn:
            btn.setText("■")
            self.currently_playing_btn = btn

    def _stop_playback(self):
        self.media_player.stop()
        if self.currently_playing_btn:
            self.currently_playing_btn.setText("▶")
        self.currently_playing_row = None
        self.currently_playing_btn = None

    def _on_media_status_changed(self, status):
        # Reset button to ▶ when playback ends naturally
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            if self.currently_playing_btn:
                self.currently_playing_btn.setText("▶")
            self.currently_playing_row = None
            self.currently_playing_btn = None

    def pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Music Folder / USB Drive")
        if folder:
            self.selected_folder = folder
            self.folder_label.setText(folder)
            self.scan_btn.setEnabled(True)

    # ------------------------------------------------------------------
    def start_scan(self):
        if not self.selected_folder or not self.classifier:
            return

        self._stop_playback()
        self.results = []
        self.results_table.setRowCount(0)
        self.dup_list.clear()
        self.export_btn.setEnabled(False)
        self.scan_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Scanning...")

        self.worker = ScanWorker(self.selected_folder, CACHE_DB_PATH, ARTIFACTS_DIR)
        self.worker.progress.connect(self.on_progress)
        self.worker.track_done.connect(self.on_track_done)
        self.worker.duplicates_found.connect(self.on_duplicates)
        self.worker.finished_scan.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop_scan(self):
        if self.worker:
            self.worker.stop()
            self.progress_label.setText("Stopping... (finishing current track)")
            self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    def on_progress(self, current, total, path):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"({current}/{total}) {os.path.basename(path)}")

    def on_track_done(self, result):
        self.results.append(result)
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)

        self.results_table.setItem(row, 1, QTableWidgetItem(os.path.basename(result.path)))

        if result.error:
            self.results_table.setItem(row, 2, QTableWidgetItem(f"ERROR: {result.error}"))
            return

        # Play/stop preview button
        play_btn = QPushButton("▶")
        play_btn.setFixedWidth(32)
        play_btn.setCheckable(False)
        play_btn.clicked.connect(lambda _, r=result, b=None: self.toggle_play(r.path, row))
        self.results_table.setCellWidget(row, 0, play_btn)

        self.results_table.setItem(row, 2, QTableWidgetItem(result.subgenre or ""))

        # Tag Genre column: shows the file's embedded genre tag.
        # Agreement with prediction -> green "✓ <tag>"; conflict -> orange tag text.
        tag_genre = getattr(result, "tag_genre", None)
        tag_match = getattr(result, "tag_match", None)
        if tag_genre:
            if tag_match is True:
                tag_item = QTableWidgetItem(f"✓ {tag_genre}")
                tag_item.setForeground(Qt.GlobalColor.darkGreen)
            elif tag_match is False:
                tag_item = QTableWidgetItem(tag_genre)
                tag_item.setForeground(Qt.GlobalColor.darkYellow)
                tag_item.setToolTip("Metadata genre disagrees with model prediction")
            else:
                tag_item = QTableWidgetItem(tag_genre)  # unmappable tag — neutral
        else:
            tag_item = QTableWidgetItem("")
        self.results_table.setItem(row, 3, tag_item)

        self.results_table.setItem(row, 4, QTableWidgetItem(result.family or ""))
        bpm_str = f"{result.bpm:.1f}" if result.bpm else ""
        self.results_table.setItem(row, 5, QTableWidgetItem(bpm_str))

        bpm_src = result.bpm_source or ""
        bpm_src_display = {"tag": "tag", "detected": "detected", "cached": ""}.get(bpm_src, bpm_src)
        self.results_table.setItem(row, 6, QTableWidgetItem(bpm_src_display))

        topk_str = ", ".join(f"{lbl} ({p:.0%})" for lbl, p in (result.top_k or []))
        self.results_table.setItem(row, 7, QTableWidgetItem(topk_str))
        self.results_table.setItem(row, 8, QTableWidgetItem(result.stage or ""))

        # Highlight low-confidence / non-EDM rows
        if result.subgenre == UNSORTED_LABEL:
            for col in range(1, 9):
                item = self.results_table.item(row, col)
                if item:
                    item.setBackground(Qt.GlobalColor.yellow)

        # Override dropdown
        original_prediction = result.subgenre
        original_confidence = (result.top_k[0][1] if result.top_k else None)
        combo = QComboBox()
        combo.addItems(ALL_OVERRIDE_OPTIONS)
        if result.subgenre in ALL_OVERRIDE_OPTIONS:
            combo.setCurrentText(result.subgenre)
        else:
            combo.addItem(result.subgenre or "")
            combo.setCurrentText(result.subgenre or "")

        def _on_override_changed(text, r=result):
            setattr(r, "subgenre", text)
            if r.features is not None:
                from feedback_store import log_correction
                log_correction(
                    path=r.path,
                    features=r.features,
                    predicted_subgenre=original_prediction,
                    predicted_family=r.family,
                    corrected_label=text,
                    confidence=original_confidence,
                )
                self.update_feedback_label()

        combo.currentTextChanged.connect(_on_override_changed)
        self.results_table.setCellWidget(row, 9, combo)

    def on_duplicates(self, dups):
        self.duplicates = dups
        self.dup_list.clear()

        exact = dups.get("exact", [])
        near = dups.get("near", [])

        if not exact and not near:
            self.dup_list.addItem("No duplicates found.")
            return

        if exact:
            self.dup_list.addItem("=== EXACT DUPLICATES (identical files) ===")
            for group in exact:
                self.dup_list.addItem("  Group:")
                for p in group:
                    self.dup_list.addItem(f"    {p}")

        if near:
            self.dup_list.addItem("")
            self.dup_list.addItem("=== POSSIBLE DUPLICATES (same track, different format/rip) ===")
            for group in near:
                self.dup_list.addItem("  Group:")
                for p in group:
                    self.dup_list.addItem(f"    {p}")

    def on_finished(self, results, was_stopped):
        if was_stopped:
            self.progress_label.setText(
                f"Stopped. Processed {len(results)} tracks before stopping "
                "(already-scanned tracks are cached)."
            )
        else:
            self.progress_label.setText(f"Done. Scanned {len(results)} tracks.")
        self.scan_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.export_btn.setEnabled(len(results) > 0)
        self.update_summary()

    def update_summary(self):
        """Show a family-level breakdown (the reliable signal) and a
        subgenre breakdown within each family (the best-guess signal)."""
        self.summary_list.clear()

        from collections import Counter, defaultdict
        family_counts = Counter()
        family_subgenres = defaultdict(Counter)

        for r in self.results:
            if r.error:
                continue
            fam = r.family or "Unknown"
            sub = r.subgenre or "Unknown"
            family_counts[fam] += 1
            family_subgenres[fam][sub] += 1

        total = sum(family_counts.values())
        self.summary_list.addItem(f"Total tracks: {total}")
        self.summary_list.addItem("")
        self.summary_list.addItem("=== By Family (reliable — recommended for crates) ===")
        for fam, count in family_counts.most_common():
            pct = 100 * count / total if total else 0
            self.summary_list.addItem(f"{fam:25s} {count:4d}  ({pct:.0f}%)")

        self.summary_list.addItem("")
        self.summary_list.addItem("=== Subgenre breakdown within each family (best-guess hint) ===")
        for fam, count in family_counts.most_common():
            self.summary_list.addItem("")
            self.summary_list.addItem(f"{fam}:")
            for sub, sc in family_subgenres[fam].most_common():
                self.summary_list.addItem(f"    {sub:25s} {sc:4d}")

    def on_error(self, msg):
        QMessageBox.critical(self, "Scan error", msg)
        self.scan_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    def export_xml(self):
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Rekordbox XML", "milkcrate_library.xml", "XML Files (*.xml)"
        )
        if not out_path:
            return

        grouping_map = {
            "Family only (recommended)": "family",
            "Family > Subgenre": "family_then_subgenre",
            "Subgenre only": "subgenre",
        }
        grouping = grouping_map[self.grouping_combo.currentText()]

        family_map = self.classifier.family_map if self.classifier else {}

        tracks = []
        for r in self.results:
            if r.error:
                continue
            # If overridden to an EDM subgenre, look up its family;
            # otherwise (non-EDM / Unsorted), subgenre == family.
            family = family_map.get(r.subgenre, r.subgenre or "Unknown")
            tracks.append({
                "path": r.path,
                "subgenre": r.subgenre,
                "family": family,
                "bpm": r.bpm,
                "duration_sec": r.duration_sec,
            })

        try:
            build_rekordbox_xml(tracks, out_path, grouping=grouping)
            QMessageBox.information(
                self, "Export complete",
                f"Saved {out_path}\n\n"
                "In Rekordbox: Preferences > Advanced > Database > "
                "Import Library, then select this file."
            )
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
