"""Locate the per-day snapshot folders stored under history/.

Each day lives in ``history/<YYYY-MM-DD>/`` and holds that day's
``headlines.json``, ``clusters.json``, and rendered ``maps/`` images.
"""

import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
HISTORY_DIR = PROJECT_DIR / "history"

SNAPSHOT_FOLDER = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
CLUSTERS_FILENAME = "clusters.json"
HEADLINES_FILENAME = "headlines.json"
MAPS_FOLDER = "maps"


def snapshot_dir(snapshot_date: str) -> Path:
    """Return the history folder for one snapshot date."""
    return HISTORY_DIR / snapshot_date


def clusters_file(snapshot_date: str) -> Path:
    return snapshot_dir(snapshot_date) / CLUSTERS_FILENAME


def headlines_file(snapshot_date: str) -> Path:
    return snapshot_dir(snapshot_date) / HEADLINES_FILENAME


def maps_dir(snapshot_date: str) -> Path:
    return snapshot_dir(snapshot_date) / MAPS_FOLDER


def _complete_snapshot_dirs(*required_files: str) -> list[Path]:
    """Return dated snapshot folders holding every required file, newest first."""
    if not HISTORY_DIR.is_dir():
        raise FileNotFoundError(f"No history folder found at {HISTORY_DIR}")
    folders = [
        path
        for path in HISTORY_DIR.iterdir()
        if path.is_dir()
        and SNAPSHOT_FOLDER.fullmatch(path.name)
        and all((path / name).is_file() for name in required_files)
    ]
    if not folders:
        required = " and ".join(required_files)
        raise FileNotFoundError(
            f"No dated snapshots with {required} found in {HISTORY_DIR}"
        )
    return sorted(folders, key=lambda path: path.name, reverse=True)


def snapshot_dirs() -> list[Path]:
    """Return complete dated snapshots (headlines + clusters), newest first."""
    return _complete_snapshot_dirs(HEADLINES_FILENAME, CLUSTERS_FILENAME)


def snapshot_dates() -> list[str]:
    """Return available snapshot dates, newest first."""
    return [path.name for path in snapshot_dirs()]


def latest_headlines_file() -> Path:
    """Return the newest snapshot's headlines file."""
    return _complete_snapshot_dirs(HEADLINES_FILENAME)[0] / HEADLINES_FILENAME
