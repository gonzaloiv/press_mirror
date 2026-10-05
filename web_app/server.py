import json
import sys
from datetime import date
from pathlib import Path

from flask import Flask, abort, render_template, request, url_for

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from shared.country_codes import (
    COUNTRY_NAMES,
    ISO_FIX,
    to_iso,
)
from web_app.map_presentation import european_coverage, map_viewport

HISTORY_DIR = PROJECT_DIR / "history"
STATIC_DIR = PROJECT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


def cluster_files() -> list[Path]:
    """Return available dated cluster snapshots, newest first."""
    files = sorted(HISTORY_DIR.glob("clusters_????-??-??.json"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No dated cluster files found in {HISTORY_DIR}")
    return files


def snapshot_file(snapshot_date: str | None = None) -> Path:
    """Find a requested snapshot or default to the most recent one."""
    files = cluster_files()
    if snapshot_date is None:
        return files[0]
    for path in files:
        if path.stem == f"clusters_{snapshot_date}":
            return path
    abort(404)


def active_countries(snapshot_date: str) -> list[str]:
    """Return countries whose feeds produced headlines in this snapshot."""
    headlines_file = HISTORY_DIR / f"headlines_{snapshot_date}.json"
    with headlines_file.open(encoding="utf-8") as f:
        headlines = json.load(f)
    return sorted({to_iso(headline["country"]) for headline in headlines})


@app.route("/")
def index():
    files = cluster_files()
    dates = [path.stem.removeprefix("clusters_") for path in files]
    selected_date = request.args.get("date", dates[0])
    if selected_date not in dates:
        abort(404)

    clusters_file = snapshot_file(selected_date)
    with clusters_file.open(encoding="utf-8") as f:
        clusters = json.load(f)

    cards = [
        {
            "index": index,
            "image_path": (
                f"/static/maps/{selected_date}/{index}_small.png"
            ),
            "map_url": url_for("map_view", cluster_id=index, date=selected_date),
        }
        for index in range(min(6, len(clusters)))
    ]
    date_index = dates.index(selected_date)
    newer_date = dates[date_index - 1] if date_index > 0 else None
    older_date = dates[date_index + 1] if date_index + 1 < len(dates) else None
    return render_template(
        "grid.html",
        cards=cards,
        date_label=f"{date.fromisoformat(selected_date).day} "
        f"{date.fromisoformat(selected_date):%b %Y}",
        selected_date=selected_date,
        newer_url=url_for("index", date=newer_date) if newer_date else None,
        older_url=url_for("index", date=older_date) if older_date else None,
    )

@app.route("/map/<int:cluster_id>")
def map_view(cluster_id):
    clusters_file = snapshot_file(request.args.get("date"))
    with clusters_file.open(encoding="utf-8") as f:
        clusters = json.load(f)

    if cluster_id >= len(clusters):
        abort(404)
    cluster = clusters[cluster_id]
    snapshot_date = date.fromisoformat(
        clusters_file.stem.removeprefix("clusters_")
    )
    available_countries = active_countries(snapshot_date.isoformat())
    thumbnail = request.args.get("thumbnail") == "1"

    # Normalize once, here. Footer still shows the original "UK" for readability.
    covered = sorted({to_iso(a["country"]) for a in cluster["articles"]})

    # Cluster snapshots retain the source article nearest the centroid.
    sources_by_country = {}
    for article in cluster["articles"]:
        source = article.get("source")
        if not source:
            continue

        country_code = to_iso(article["country"])
        country_sources = sources_by_country.setdefault(country_code, {})
        country_sources.setdefault(
            source,
            {
                "url": article.get("link") or None,
                "headline": article.get("title", ""),
            },
        )

    source_groups = [
        {
            "name": COUNTRY_NAMES.get(country_code, country_code),
            "outlets": [
                {"name": source, **article}
                for source, article in sorted(
                    country_sources.items(), key=lambda outlet: outlet[0].casefold()
                )
            ],
        }
        for country_code, country_sources in sources_by_country.items()
    ]
    source_groups.sort(
        key=lambda group: (-len(group["outlets"]), group["name"].casefold())
    )

    return render_template(
        "map.html",
        label=cluster["label"],
        thumbnail=thumbnail,
        covered=covered,
        map_bounds=map_viewport(covered),
        coverage_copy=european_coverage(
            covered, max_countries=5 if thumbnail else None
        ),
        snapshot_date=snapshot_date.isoformat(),
        date_label=f"{snapshot_date.day} {snapshot_date:%b %Y}",
        countries=available_countries,
        source_groups=source_groups,
        iso_fix=ISO_FIX,
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
