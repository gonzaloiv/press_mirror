import json
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

INPUT_FILE = "embeddings.npz"
OUTPUT_FILE = "clusters.json"

SIMILARITY_THRESHOLD = 0.75    # tune this — see notes below


def load_data():
    data = np.load(INPUT_FILE, allow_pickle=True)
    return (
        data["embeddings"],
        list(data["titles"]),
        list(data["sources"]),
        list(data["countries"]),
        list(data["links"]),
    )


def cluster(embeddings):
    """Group indices into clusters. Uses cosine distance, not raw vectors."""
    # AgglomerativeClustering needs a *distance* threshold, not a similarity one.
    # cosine distance = 1 - cosine similarity, so 0.75 similarity -> 0.25 distance.
    distance_threshold = 1.0 - SIMILARITY_THRESHOLD

    model = AgglomerativeClustering(
        n_clusters=None,                    # let the algorithm decide
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",                  # average distance between groups
    )
    return model.fit_predict(embeddings)


def build_clusters(labels, titles, sources, countries, links, embeddings):
    """Turn flat labels into structured cluster dicts with similarity stats."""
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)

    result = []
    for label, indices in clusters.items():
        if len(indices) < 2:
            continue                        # skip singletons for the MVP

        # Recompute pairwise similarity *within* this cluster for the display.
        sub = embeddings[indices]
        sim_matrix = cosine_similarity(sub)
        # Average of the off-diagonal (exclude self-similarity = 1.0).
        n = len(indices)
        avg_sim = (sim_matrix.sum() - n) / (n * n - n)

        articles = []
        for i in indices:
            articles.append({
                "title": titles[i],
                "source": sources[i],
                "country": countries[i],
                "link": links[i],
            })

        result.append({
            "label": titles[indices[0]],    # topic label = first headline (placeholder)
            "size": len(indices),
            "avg_similarity": round(float(avg_sim), 3),
            "articles": articles,
        })

    # Show the most-covered stories first.
    result.sort(key=lambda c: c["size"], reverse=True)
    return result


def main():
    embeddings, titles, sources, countries, links = load_data()
    print(f"Loaded {len(titles)} headlines.")

    labels = cluster(embeddings)
    print(f"Found {len(set(labels))} raw clusters.")

    clusters = build_clusters(labels, titles, sources, countries, links, embeddings)
    print(f"Kept {len(clusters)} clusters with 2+ articles.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"Written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()