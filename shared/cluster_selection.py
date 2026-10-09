"""Shared cluster-selection rules for the web app and image renderer."""

from shared.country_codes import to_iso

MIN_CLUSTER_SIMILARITY = 0.74
MIN_CLUSTER_REPRESENTATIVE_SIMILARITY = 0.72
MIN_CLUSTER_SOURCES = 6
MIN_CLUSTER_COUNTRIES = 4


def selected_cluster_indices(
    clusters: list[dict],
    *,
    min_similarity: float = MIN_CLUSTER_SIMILARITY,
    min_representative_similarity: float = (
        MIN_CLUSTER_REPRESENTATIVE_SIMILARITY
    ),
    min_sources: int = MIN_CLUSTER_SOURCES,
    min_countries: int = MIN_CLUSTER_COUNTRIES,
) -> list[int]:
    """Return clusters meeting cohesion, similarity, and coverage minimums."""
    selected = []
    for index, cluster in enumerate(clusters):
        if cluster["avg_similarity"] < min_similarity:
            continue
        representative_similarity = cluster.get(
            "min_representative_similarity",
            cluster["avg_similarity"],
        )
        if representative_similarity < min_representative_similarity:
            continue
        if len({
            article["source"]
            for article in cluster["articles"]
            if article.get("source")
        }) < min_sources:
            continue
        if len({
            to_iso(article["country"])
            for article in cluster["articles"]
        }) < min_countries:
            continue
        selected.append(index)
    return selected
