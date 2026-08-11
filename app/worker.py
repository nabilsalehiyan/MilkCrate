"""
worker.py
=========
QThread worker that runs the library scan in the background, emitting
progress signals so the GUI stays responsive. Supports being stopped
mid-scan via stop().
"""

from __future__ import annotations

import threading

from PyQt6.QtCore import QThread, pyqtSignal

from scanner import find_audio_files, scan_library, detect_duplicates
from cache_db import TrackCache
from classifier import TwoStageClassifier


class ScanWorker(QThread):
    progress = pyqtSignal(int, int, str)   # current, total, current_path
    track_done = pyqtSignal(object)        # ScanResult
    finished_scan = pyqtSignal(list, bool) # list of ScanResult, was_stopped
    duplicates_found = pyqtSignal(dict)    # {"exact": [...], "near": [...]}
    error = pyqtSignal(str)

    def __init__(self, root: str, cache_path: str, model_artifacts_dir: str):
        super().__init__()
        self.root = root
        self.cache_path = cache_path
        self.model_artifacts_dir = model_artifacts_dir
        self._stop_event = threading.Event()

    def stop(self):
        """Request the scan to stop after the current track finishes."""
        self._stop_event.set()

    def run(self):
        try:
            cache = TrackCache(self.cache_path)
            classifier = TwoStageClassifier(self.model_artifacts_dir)

            paths = find_audio_files(self.root)
            total = len(paths)

            results = []
            for i, result in enumerate(
                scan_library(self.root, cache, classifier, stop_event=self._stop_event),
                start=1,
            ):
                results.append(result)
                self.progress.emit(i, total, result.path)
                self.track_done.emit(result)

            was_stopped = self._stop_event.is_set()

            dups = {"exact": [], "near": []}
            if results:
                dups = detect_duplicates(self.root, cache)

            cache.close()

            self.duplicates_found.emit(dups)
            self.finished_scan.emit(results, was_stopped)

        except Exception as e:
            self.error.emit(str(e))
