import json
from pathlib import Path

from flask import Flask, render_template

app = Flask(__name__)
HISTORY_DIR = Path(__file__).resolve().parent / "history"


def latest_clusters_file() -> Path:
    """Find the most recent dated cluster snapshot."""
    files = sorted(HISTORY_DIR.glob("clusters_????-??-??.json"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No dated cluster files found in {HISTORY_DIR}")
    return files[0]


def similarity_band(score: float) -> tuple[str, str]:
    """Turn a raw score into a label + CSS class."""
    if score >= 0.8:
        return "High", "high"
    if score >= 0.7:
        return "Medium", "medium"
    return "Low", "low"


@app.route("/")
def index():
    with latest_clusters_file().open(encoding="utf-8") as f:
        clusters = json.load(f)

    # Enrich each cluster with band labels for display.
    for c in clusters:
        band, css = similarity_band(c["avg_similarity"])
        c["band"] = band
        c["band_css"] = css

    return render_template("index.html", clusters=clusters)

ISO_FIX = {
    "UK": "GB",
}

@app.route("/map/<int:cluster_id>")
def map_view(cluster_id):
    with latest_clusters_file().open(encoding="utf-8") as f:
        clusters = json.load(f)

    cluster = clusters[cluster_id]

    # Normalize once, here. Footer still shows the original "UK" for readability.
    covered = sorted({ISO_FIX.get(a["country"], a["country"]) for a in cluster["articles"]})

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
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    
    from flask import Flask, render_template
import json
