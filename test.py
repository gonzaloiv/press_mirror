# In your terminal or a scratch script
import json
with open("clusters.json", encoding="utf-8") as f:
    clusters = json.load(f)
c = clusters[1]
print("Countries in cluster:", sorted({a["country"] for a in c["articles"]}))
print("Sources:", [(a["country"], a["source"]) for a in c["articles"]])