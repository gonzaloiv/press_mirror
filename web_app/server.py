import json
import sys
from datetime import date
from pathlib import Path

from flask import Flask, render_template

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from shared.country_codes import (
    COUNTRY_NAMES,
    ISO_FIX,
    NON_EUROPEAN_COUNTRIES,
    to_iso,
)

HISTORY_DIR = PROJECT_DIR / "history"
STATIC_DIR = PROJECT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


def latest_clusters_file() -> Path:
    """Find the most recent dated cluster snapshot."""
    files = sorted(HISTORY_DIR.glob("clusters_????-??-??.json"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No dated cluster files found in {HISTORY_DIR}")
    return files[0]


def active_countries(snapshot_date: str) -> list[str]:
    """Return countries whose feeds produced headlines in this snapshot."""
    headlines_file = HISTORY_DIR / f"headlines_{snapshot_date}.json"
    with headlines_file.open(encoding="utf-8") as f:
        headlines = json.load(f)
    return sorted({to_iso(headline["country"]) for headline in headlines})


def european_coverage(countries: list[str]) -> str:
    """Format country codes as a concise, naturally sorted coverage subtitle."""
    normalized_countries = {to_iso(code) for code in countries}
    names = sorted(
        (COUNTRY_NAMES.get(code, code) for code in normalized_countries),
        key=lambda name: name.removeprefix("the ").casefold(),
    )
    if not names:
        return "No coverage found."

    country_count = len(names)
    display_names = names
    if country_count > 5:
        display_names = names[:4] + [f"{country_count - 4} others"]

    if len(display_names) == 1:
        country_list = display_names[0]
    elif len(display_names) == 2:
        country_list = " and ".join(display_names)
    else:
        country_list = f"{', '.join(display_names[:-1])}, and {display_names[-1]}"

    is_international = bool(normalized_countries & NON_EUROPEAN_COUNTRIES)
    prefix = (
        "Coverage from"
        if is_international or country_count == 1
        else "European coverage from"
    )
    return f"{prefix} {country_list}."


@app.route("/")
def index():
    clusters_file = latest_clusters_file()
    with clusters_file.open(encoding="utf-8") as f:
        clusters = json.load(f)

    snapshot_date = clusters_file.stem.removeprefix("clusters_")
    cards = [
        {
            "index": index,
            "countries": sorted({
                to_iso(article["country"])
                for article in cluster["articles"]
            }),
            "image_path": (
                f"/static/maps/{snapshot_date}/{index}_small.png"
            ),
        }
        for index, cluster in enumerate(clusters[:6])
    ]
    return render_template(
        "grid.html",
        cards=cards,
    )

@app.route("/map/<int:cluster_id>")
def map_view(cluster_id):
    clusters_file = latest_clusters_file()
    with clusters_file.open(encoding="utf-8") as f:
        clusters = json.load(f)

    cluster = clusters[cluster_id]
    snapshot_date = date.fromisoformat(
        clusters_file.stem.removeprefix("clusters_")
    )
    available_countries = active_countries(snapshot_date.isoformat())

    # Normalize once, here. Footer still shows the original "UK" for readability.
    covered = sorted({to_iso(a["country"]) for a in cluster["articles"]})

    # Dedupe sources per country for the footer.
    sources_by_country = {}
    for a in cluster["articles"]:
        sources_by_country.setdefault(a["country"], set()).add(a["source"])
    sources_by_country = {k: sorted(v) for k, v in sorted(sources_by_country.items())}

    return render_template(
        "map.html",
        label=cluster["label"],
        covered=covered,
        coverage_copy=european_coverage(covered),
        snapshot_date=snapshot_date.isoformat(),
        date_label=f"{snapshot_date.day} {snapshot_date:%b %Y}",
        countries=available_countries,
        sources_by_country=sources_by_country,
        iso_fix=ISO_FIX,
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    
    from flask import Flask, render_template
import json
