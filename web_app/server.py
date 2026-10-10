import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from flask import Flask, abort, render_template, request, send_from_directory, url_for

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from shared.country_codes import (
    COUNTRY_NAMES,
    EUROPEAN_COUNTRIES,
    ISO_FIX,
    to_iso,
)
from shared import snapshots
from shared.cluster_selection import selected_cluster_indices
from web_app.map_presentation import european_coverage, map_viewport

STATIC_DIR = PROJECT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
SOURCES_FILE = PROJECT_DIR / "pipeline" / "sources.json"

APP_NAME = r"Press\Mirror"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.jinja_env.globals["app_name"] = APP_NAME


def latest_fetched_at(articles: list[dict]) -> datetime | None:
    timestamps = []
    for article in articles:
        fetched_at = article.get("fetched_at")
        if not fetched_at:
            continue
        timestamp = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        timestamps.append(timestamp.astimezone(timezone.utc))
    return max(timestamps, default=None)


def resolve_snapshot_date(snapshot_date: str | None) -> str:
    """Validate a requested snapshot date or default to the most recent one."""
    dates = snapshots.snapshot_dates()
    if snapshot_date is None:
        return dates[0]
    if snapshot_date not in dates:
        abort(404)
    return snapshot_date


def active_countries(snapshot_date: str) -> list[str]:
    """Return countries whose feeds produced headlines in this snapshot."""
    with snapshots.headlines_file(snapshot_date).open(encoding="utf-8") as f:
        headlines = json.load(f)
    return sorted({to_iso(headline["country"]) for headline in headlines})


def source_groups() -> list[dict]:
    """Group the configured feeds by country for the about page."""
    with SOURCES_FILE.open(encoding="utf-8") as f:
        sources = json.load(f)
    outlets_by_country: dict[str, list[dict]] = {}
    for feed in sources["feeds"]:
        country = to_iso(feed["country"])
        outlets_by_country.setdefault(country, []).append(
            {"name": feed["source"], "url": feed["site"]}
        )
    groups = [
        {
            "name": COUNTRY_NAMES.get(country, country).removeprefix("the "),
            "outlets": sorted(
                outlets, key=lambda outlet: outlet["name"].casefold()
            ),
        }
        for country, outlets in outlets_by_country.items()
    ]
    groups.sort(key=lambda group: group["name"].casefold())
    return groups


@app.route("/")
def index():
    dates = snapshots.snapshot_dates()
    selected_date = request.args.get("date", dates[0])
    if selected_date not in dates:
        abort(404)

    with snapshots.clusters_file(selected_date).open(encoding="utf-8") as f:
        clusters = json.load(f)

    headlines_path = snapshots.headlines_file(selected_date)
    headlines = []
    if headlines_path.exists():
        with headlines_path.open(encoding="utf-8") as f:
            headlines = json.load(f)
    ingestion_time = latest_fetched_at(headlines)

    cards = []
    for index in selected_cluster_indices(clusters):
        cluster = clusters[index]
        source_names = sorted(
            {
                article["source"]
                for article in cluster["articles"]
                if article.get("source")
            },
            key=str.casefold,
        )
        country_count = len(
            {to_iso(article["country"]) for article in cluster["articles"]}
        )
        cards.append(
            {
                "index": index,
                "image_path": url_for(
                    "snapshot_map",
                    snapshot_date=selected_date,
                    filename=f"{index}_small.png",
                ),
                "map_url": url_for(
                    "detail_view", cluster_id=index, date=selected_date
                ),
                "label": cluster["label"],
                "source_names": source_names,
                "outlet_count": len(source_names),
                "country_count": country_count,
            }
        )
    date_index = dates.index(selected_date)
    newer_date = dates[date_index - 1] if date_index > 0 else None
    older_date = dates[date_index + 1] if date_index + 1 < len(dates) else None
    return render_template(
        "home.html",
        cards=cards,
        date_label=f"{date.fromisoformat(selected_date).day} "
        f"{date.fromisoformat(selected_date):%b %Y}",
        selected_date=selected_date,
        ingestion_time=(
            ingestion_time.isoformat() if ingestion_time else None
        ),
        ingestion_time_label=(
            ingestion_time.strftime("%H:%M UTC") if ingestion_time else None
        ),
        newer_url=url_for("index", date=newer_date) if newer_date else None,
        older_url=url_for("index", date=older_date) if older_date else None,
    )


@app.route("/about")
def about():
    groups = source_groups()
    return render_template(
        "about.html",
        source_groups=groups,
        outlet_count=sum(len(group["outlets"]) for group in groups),
        country_count=len(groups),
    )


@app.route("/maps/<snapshot_date>/<path:filename>")
def snapshot_map(snapshot_date, filename):
    """Serve rendered map images from a snapshot's history folder."""
    return send_from_directory(snapshots.maps_dir(snapshot_date), filename)


@app.route("/map/<int:cluster_id>")
def detail_view(cluster_id):
    snapshot_date = resolve_snapshot_date(request.args.get("date"))
    with snapshots.clusters_file(snapshot_date).open(encoding="utf-8") as f:
        clusters = json.load(f)

    if cluster_id >= len(clusters):
        abort(404)
    cluster = clusters[cluster_id]
    snapshot_day = date.fromisoformat(snapshot_date)
    available_countries = active_countries(snapshot_date)
    thumbnail = request.args.get("thumbnail") == "1"

    # Normalize once, here. Footer still shows the original "UK" for readability.
    covered = sorted({to_iso(a["country"]) for a in cluster["articles"]})
    tracked = sorted({to_iso(country) for country in available_countries})
    covered_country_names = sorted(
        (
            COUNTRY_NAMES.get(country, country).removeprefix("the ")
            for country in covered
        ),
        key=str.casefold,
    )
    not_covered_country_names = sorted(
        (
            COUNTRY_NAMES.get(country, country).removeprefix("the ")
            for country in set(tracked) - set(covered)
        ),
        key=str.casefold,
    )

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
        "card.html" if thumbnail else "detail.html",
        label=cluster["label"],
        terms=cluster.get("terms", []),
        thumbnail=thumbnail,
        covered=covered,
        covered_country_names=covered_country_names,
        not_covered_country_names=not_covered_country_names,
        map_bounds=map_viewport(covered),
        coverage_copy=european_coverage(
            covered, max_countries=5 if thumbnail else None
        ),
        source_names=sorted(
            {
                article["source"]
                for article in cluster["articles"]
                if article.get("source")
            },
            key=str.casefold,
        ),
        outlet_count=len(
            {
                article["source"]
                for article in cluster["articles"]
                if article.get("source")
            }
        ),
        country_count=len(
            {to_iso(article["country"]) for article in cluster["articles"]}
        ),
        snapshot_date=snapshot_date,
        date_label=f"{snapshot_day.day} {snapshot_day:%b %Y}",
        countries=available_countries,
        european_countries=sorted(EUROPEAN_COUNTRIES),
        source_groups=source_groups,
        iso_fix=ISO_FIX,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
