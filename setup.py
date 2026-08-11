"""
py2app setup script for MilkCrate DJ
Run: python3 setup.py py2app
"""
from setuptools import setup

APP = ['app/main.py']
DATA_FILES = [
    ('model/artifacts', [
        'model/artifacts/genre_model.joblib',
        'model/artifacts/label_encoder.joblib',
        'model/artifacts/family_map.json',
        'model/artifacts/feature_columns.json',
    ]),
    ('app', [
        'app/cache_db.py',
        'app/classifier.py',
        'app/coarse_classifier.py',
        'app/duplicate_detect.py',
        'app/feature_extract.py',
        'app/feedback_store.py',
        'app/rekordbox_export.py',
        'app/scanner.py',
        'app/tag_reader.py',
        'app/worker.py',
    ]),
]

OPTIONS = {
    'argv_emulation': False,
    'packages': [
        'numpy', 'pandas', 'sklearn', 'lightgbm', 'librosa',
        'soundfile', 'mutagen', 'joblib', 'PyQt6', 'pyarrow',
        'scipy', 'audioread', 'decorator', 'lazy_loader',
        'msgpack', 'pooch', 'platformdirs', 'packaging',
    ],
    'includes': [
        'PyQt6.QtCore', 'PyQt6.QtWidgets', 'PyQt6.QtMultimedia',
        'PyQt6.QtGui', 'sklearn.utils._cython_blas',
        'sklearn.neighbors._partition_nodes',
        'sklearn.tree._utils',
    ],
    'excludes': ['tkinter', 'matplotlib', 'IPython', 'jupyter'],
    'iconfile': None,
    'plist': {
        'CFBundleName': 'MilkCrate DJ',
        'CFBundleDisplayName': 'MilkCrate DJ',
        'CFBundleIdentifier': 'com.milkcrate.dj',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': True,
        'NSMicrophoneUsageDescription': 'MilkCrate DJ needs microphone access for audio playback.',
    },
}

setup(
    app=APP,
    name='MilkCrate DJ',
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
