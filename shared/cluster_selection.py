"""Shared cluster-selection rules for the web app and image renderer."""

from shared.country_codes import to_iso

MIN_CLUSTER_SIMILARITY = 0.72
MIN_CLUSTER_REPRESENTATIVE_SIMILARITY = 0.68
MIN_CLUSTER_SOURCES = 6
MIN_CLUSTER_COUNTRIES = 4


def selected_cluster_indices(clusters: list[dict]) -> list[int]:
    """Return indices of clusters meeting the similarity and coverage minimums."""
    return [
        index
        for index, cluster in enumerate(clusters)
        if (
            cluster["avg_similarity"] >= MIN_CLUSTER_SIMILARITY
            and cluster.get(
                "min_representative_similarity", cluster["avg_similarity"]
            ) >= MIN_CLUSTER_REPRESENTATIVE_SIMILARITY
            and len({
                article["source"]
                for article in cluster["articles"]
                if article.get("source")
            }) >= MIN_CLUSTER_SOURCES
            and len({
                to_iso(article["country"])
                for article in cluster["articles"]
            }) >= MIN_CLUSTER_COUNTRIES
        )
    ]
