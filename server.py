import json
from flask import Flask, render_template

app = Flask(__name__)
CLUSTERS_FILE = "clusters.json"


def similarity_band(score: float) -> tuple[str, str]:
    """Turn a raw score into a label + CSS class."""
    if score >= 0.85:
        return "High", "high"
    if score >= 0.75:
        return "Medium", "medium"
    return "Low", "low"


@app.route("/")
def index():
    with open(CLUSTERS_FILE, encoding="utf-8") as f:
        clusters = json.load(f)

    # Enrich each cluster with band labels for display.
    for c in clusters:
        band, css = similarity_band(c["avg_similarity"])
        c["band"] = band
        c["band_css"] = css

    return render_template("index.html", clusters=clusters)


if __name__ == "__main__":
    app.run(debug=True, port=5000)