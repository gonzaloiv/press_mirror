import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from flask import Flask, abort, g, render_template, request, url_for

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from shared.country_codes import (
    EUROPEAN_COUNTRIES,
    ISO_FIX,
    country_name,
    to_iso,
)
from shared.cluster_selection import selected_cluster_indices
from shared.editions import EDITIONS, DEFAULT_EDITION_KEY, edition_from_path
from web_app.i18n import ABOUT_TITLE, about_sections, coverage_line, ui_strings
from web_app.map_presentation import european_coverage, map_viewport

STATIC_DIR = PROJECT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

APP_NAME = r"Press\Mirror"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.jinja_env.globals["app_name"] = APP_NAME


def eurl(endpoint: str, **values) -> str:
    """Build a URL for the edition currently being rendered."""
    edition = getattr(g, "edition", EDITIONS[DEFAULT_EDITION_KEY])
    return url_for(f"{endpoint}_{edition.key}", **values)


app.jinja_env.globals["eurl"] = eurl


@app.before_request
def resolve_edition() -> None:
    g.edition = edition_from_path(request.path)


@app.context_processor
def inject_i18n() -> dict:
    edition = getattr(g, "edition", EDITIONS[DEFAULT_EDITION_KEY])
    other_key = "es" if edition.key == "en" else "en"
    strings = ui_strings(edition.language, APP_NAME)
    return {
        "strings": strings,
        "edition": edition,
        "language": edition.language,
        "other_edition_url": url_for(f"index_{other_key}"),
        "language_switch_label": strings["language_switch"],
        "language_switch_aria": strings["language_switch_aria"],
    }


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


def cluster_files(edition) -> list[Path]:
    """Return available dated cluster snapshots, newest first."""
    files = sorted(
        edition.history_dir.glob("clusters_????-??-??.json"), reverse=True
    )
    if not files:
        raise FileNotFoundError(
            f"No dated cluster files found in {edition.history_dir}"
        )
    return files


def snapshot_file(edition, snapshot_date: str | None = None) -> Path:
    """Find a requested snapshot or default to the most recent one."""
    files = cluster_files(edition)
    if snapshot_date is None:
        return files[0]
    for path in files:
        if path.stem == f"clusters_{snapshot_date}":
            return path
    abort(404)


def active_countries(edition, snapshot_date: str) -> list[str]:
    """Return countries whose feeds produced headlines in this snapshot."""
    headlines_file = edition.history_dir / f"headlines_{snapshot_date}.json"
    with headlines_file.open(encoding="utf-8") as f:
        headlines = json.load(f)
    return sorted({to_iso(headline["country"]) for headline in headlines})


def source_groups(edition) -> list[dict]:
    """Group the configured feeds by country for the about page."""
    with edition.sources_file.open(encoding="utf-8") as f:
        sources = json.load(f)
    outlets_by_country: dict[str, list[dict]] = {}
    for feed in sources["feeds"]:
        country = to_iso(feed["country"])
        outlets_by_country.setdefault(country, []).append(
            {"name": feed["source"], "url": feed["site"]}
        )
    groups = [
        {
            "name": country_name(country, edition.language).removeprefix("the "),
            "outlets": sorted(
                outlets, key=lambda outlet: outlet["name"].casefold()
            ),
        }
        for country, outlets in outlets_by_country.items()
    ]
    groups.sort(key=lambda group: group["name"].casefold())
    return groups


def render_index(edition):
    files = cluster_files(edition)
    dates = [path.stem.removeprefix("clusters_") for path in files]
    selected_date = request.args.get("date", dates[0])
    if selected_date not in dates:
        abort(404)

    clusters_file = snapshot_file(edition, selected_date)
    with clusters_file.open(encoding="utf-8") as f:
        clusters = json.load(f)

    headlines_file = edition.history_dir / f"headlines_{selected_date}.json"
    headlines = []
    if headlines_file.exists():
        with headlines_file.open(encoding="utf-8") as f:
            headlines = json.load(f)
    ingestion_time = latest_fetched_at(headlines)

    cards = []
    selected = selected_cluster_indices(
        clusters,
        min_similarity=edition.min_similarity,
        min_sources=edition.min_sources,
        min_countries=edition.min_countries,
    )
    for index in selected:
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
                "image_path": (
                    f"{edition.maps_url}/{selected_date}/{index}_small.png"
                ),
                "map_url": eurl(
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
        newer_url=eurl("index", date=newer_date) if newer_date else None,
        older_url=eurl("index", date=older_date) if older_date else None,
    )


def render_about(edition):
    groups = source_groups(edition)
    outlet_count = sum(len(group["outlets"]) for group in groups)
    country_count = len(groups)
    sources_html = render_template(
        "_about_sources.html", source_groups=groups
    )
    return render_template(
        "about.html",
        source_groups=groups,
        outlet_count=outlet_count,
        country_count=country_count,
        about_title=ABOUT_TITLE.get(edition.language, ABOUT_TITLE["en"]).replace(
            "{app}", APP_NAME
        ),
        about_sections=about_sections(
            edition.language,
            APP_NAME,
            outlet_count,
            country_count,
            sources_html,
        ),
    )


def render_detail(edition, cluster_id):
    clusters_file = snapshot_file(edition, request.args.get("date"))
    with clusters_file.open(encoding="utf-8") as f:
        clusters = json.load(f)

    if cluster_id >= len(clusters):
        abort(404)
    cluster = clusters[cluster_id]
    snapshot_date = date.fromisoformat(
        clusters_file.stem.removeprefix("clusters_")
    )
    available_countries = active_countries(edition, snapshot_date.isoformat())
    thumbnail = request.args.get("thumbnail") == "1"

    # Normalize once, here. Footer still shows the original "UK" for readability.
    covered = sorted({to_iso(a["country"]) for a in cluster["articles"]})
    tracked = sorted({to_iso(country) for country in available_countries})
    covered_country_names = sorted(
        (
            country_name(country, edition.language).removeprefix("the ")
            for country in covered
        ),
        key=str.casefold,
    )
    not_covered_country_names = sorted(
        (
            country_name(country, edition.language).removeprefix("the ")
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

    grouped_sources = [
        {
            "name": country_name(country_code, edition.language),
            "outlets": [
                {"name": source, **article}
                for source, article in sorted(
                    country_sources.items(), key=lambda outlet: outlet[0].casefold()
                )
            ],
        }
        for country_code, country_sources in sources_by_country.items()
    ]
    grouped_sources.sort(
        key=lambda group: (-len(group["outlets"]), group["name"].casefold())
    )

    outlet_count = len(
        {
            article["source"]
            for article in cluster["articles"]
            if article.get("source")
        }
    )
    country_count = len(
        {to_iso(article["country"]) for article in cluster["articles"]}
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
            covered,
            max_countries=5 if thumbnail else None,
            language=edition.language,
        ),
        source_names=sorted(
            {
                article["source"]
                for article in cluster["articles"]
                if article.get("source")
            },
            key=str.casefold,
        ),
        outlet_count=outlet_count,
        country_count=country_count,
        source_meta_label=coverage_line(
            outlet_count, country_count, edition.language
        ),
        snapshot_date=snapshot_date.isoformat(),
        date_label=f"{snapshot_date.day} {snapshot_date:%b %Y}",
        countries=available_countries,
        european_countries=sorted(EUROPEAN_COUNTRIES),
        source_groups=grouped_sources,
        iso_fix=ISO_FIX,
    )


def register_edition_routes(edition) -> None:
    prefix = edition.url_prefix

    def index():
        return render_index(edition)

    def about():
        return render_about(edition)

    def detail_view(cluster_id):
        return render_detail(edition, cluster_id)

    app.add_url_rule(f"{prefix}/", endpoint=f"index_{edition.key}", view_func=index)
    app.add_url_rule(
        f"{prefix}/about", endpoint=f"about_{edition.key}", view_func=about
    )
    app.add_url_rule(
        f"{prefix}/map/<int:cluster_id>",
        endpoint=f"detail_view_{edition.key}",
        view_func=detail_view,
    )


for edition in EDITIONS.values():
    register_edition_routes(edition)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
