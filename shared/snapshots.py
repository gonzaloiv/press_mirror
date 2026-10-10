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


def snapshot_dirs() -> list[Path]:
    """Return dated folders holding clusters, newest first."""
    if not HISTORY_DIR.is_dir():
        raise FileNotFoundError(f"No history folder found at {HISTORY_DIR}")
    folders = [
        path
        for path in HISTORY_DIR.iterdir()
        if path.is_dir()
        and SNAPSHOT_FOLDER.fullmatch(path.name)
        and (path / CLUSTERS_FILENAME).is_file()
    ]
    if not folders:
        raise FileNotFoundError(f"No dated cluster snapshots found in {HISTORY_DIR}")
    return sorted(folders, key=lambda path: path.name, reverse=True)


def snapshot_dates() -> list[str]:
    """Return available snapshot dates, newest first."""
    return [path.name for path in snapshot_dirs()]


def latest_headlines_file() -> Path:
    """Return the newest snapshot's headlines file."""
    if not HISTORY_DIR.is_dir():
        raise FileNotFoundError(f"No history folder found at {HISTORY_DIR}")
    folders = sorted(
        (
            path
            for path in HISTORY_DIR.iterdir()
            if path.is_dir()
            and SNAPSHOT_FOLDER.fullmatch(path.name)
            and (path / HEADLINES_FILENAME).is_file()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    if not folders:
        raise FileNotFoundError(f"No dated headline files found in {HISTORY_DIR}")
    return folders[0] / HEADLINES_FILENAME
