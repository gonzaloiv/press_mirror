"""Edition configuration shared by the generation pipeline and the web app.

An edition is one language front: its own sources file, its own dated data
directories, its own display thresholds, and its own URL prefix. The default
English edition keeps the original flat layout and unprefixed routes; other
editions nest their data and sit behind their URL prefix.
"""

from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Edition:
    key: str
    language: str
    url_prefix: str
    sources_file: Path
    history_dir: Path
    embeddings_file: Path
    maps_dir: Path
    maps_url: str
    min_similarity: float
    min_sources: int
    min_countries: int


EDITIONS: dict[str, Edition] = {
    "en": Edition(
        key="en",
        language="en",
        url_prefix="",
        sources_file=PROJECT_DIR / "pipeline" / "sources.json",
        history_dir=PROJECT_DIR / "history",
        embeddings_file=PROJECT_DIR / "data" / "embeddings.npz",
        maps_dir=PROJECT_DIR / "static" / "maps",
        maps_url="/static/maps",
        min_similarity=0.74,
        min_sources=6,
        min_countries=4,
    ),
    "es": Edition(
        key="es",
        language="es",
        url_prefix="/es",
        sources_file=PROJECT_DIR / "pipeline" / "sources.es.json",
        history_dir=PROJECT_DIR / "history" / "es",
        embeddings_file=PROJECT_DIR / "data" / "es" / "embeddings.npz",
        maps_dir=PROJECT_DIR / "static" / "maps" / "es",
        maps_url="/static/maps/es",
        min_similarity=0.74,
        min_sources=3,
        min_countries=1,
    ),
}

DEFAULT_EDITION_KEY = "en"


def get_edition(key: str) -> Edition:
    """Return the configured edition, or raise a clear error for a bad key."""
    try:
        return EDITIONS[key]
    except KeyError:
        known = ", ".join(sorted(EDITIONS))
        raise KeyError(f"Unknown edition {key!r}. Known editions: {known}.")


def edition_from_path(path: str, default: str = DEFAULT_EDITION_KEY) -> Edition:
    """Resolve the edition whose URL prefix matches a request path."""
    for edition in EDITIONS.values():
        if edition.url_prefix and (
            path == edition.url_prefix or path.startswith(edition.url_prefix + "/")
        ):
            return edition
    return EDITIONS[default]
