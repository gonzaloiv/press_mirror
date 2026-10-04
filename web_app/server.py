import json
import sys
from pathlib import Path

from flask import Flask, render_template

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from country_codes import ISO_FIX, to_iso

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


@app.route("/")
def index():
    clusters_file = latest_clusters_file()
    with clusters_file.open(encoding="utf-8") as f:
        clusters = json.load(f)

    for c in clusters:
        c["countries"] = sorted({to_iso(a["country"]) for a in c["articles"]})

    snapshot_date = clusters_file.stem.removeprefix("clusters_")
    return render_template(
        "index.html",
        clusters=clusters,
        maps_date=snapshot_date,
    )

@app.route("/map/<int:cluster_id>")
def map_view(cluster_id):
    with latest_clusters_file().open(encoding="utf-8") as f:
        clusters = json.load(f)

    cluster = clusters[cluster_id]

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
        countries=["FR", "ES", "DE", "IT", "GB", "US", "QA", "NL", "PT", "PL"],
        sources_by_country=sources_by_country,
        iso_fix=ISO_FIX,
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    
    from flask import Flask, render_template
import json
