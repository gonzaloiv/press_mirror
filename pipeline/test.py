# In your terminal or a scratch script
import json
from pathlib import Path

history_dir = Path(__file__).resolve().parent.parent / "history"
cluster_files = sorted(history_dir.glob("clusters_????-??-??.json"), reverse=True)
if not cluster_files:
    raise FileNotFoundError(f"No dated cluster files found in {history_dir}")

with cluster_files[0].open(encoding="utf-8") as f:
    clusters = json.load(f)
c = clusters[1]
print("Countries in cluster:", sorted({a["country"] for a in c["articles"]}))
print("Sources:", [(a["country"], a["source"]) for a in c["articles"]])