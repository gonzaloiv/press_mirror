# In your terminal or a scratch script
import json
from pathlib import Path

history_dir = Path(__file__).resolve().parent.parent / "history"
snapshot_dirs = sorted(
    (
        path
        for path in history_dir.iterdir()
        if path.is_dir() and (path / "clusters.json").is_file()
    ),
    key=lambda path: path.name,
    reverse=True,
)
if not snapshot_dirs:
    raise FileNotFoundError(f"No dated cluster snapshots found in {history_dir}")

with (snapshot_dirs[0] / "clusters.json").open(encoding="utf-8") as f:
    clusters = json.load(f)
c = clusters[1]
print("Countries in cluster:", sorted({a["country"] for a in c["articles"]}))
print("Sources:", [(a["country"], a["source"]) for a in c["articles"]])