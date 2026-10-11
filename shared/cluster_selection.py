"""Shared cluster-selection rules for the web app and image renderer."""

from shared.country_codes import to_iso

MIN_CLUSTER_SIMILARITY = 0.725
MIN_CLUSTER_REPRESENTATIVE_SIMILARITY = 0.65
MIN_CLUSTER_SOURCES = 6
MIN_CLUSTER_COUNTRIES = 4
TOP_CLUSTER_COUNT = 5
MAX_CLUSTER_COUNT = 9


def _meets_coverage_gates(cluster: dict) -> bool:
    sources = {
        article["source"]
        for article in cluster["articles"]
        if article.get("source")
    }
    countries = {to_iso(article["country"]) for article in cluster["articles"]}
    return (
        len(sources) >= MIN_CLUSTER_SOURCES
        and len(countries) >= MIN_CLUSTER_COUNTRIES
        and cluster.get(
            "min_representative_similarity", cluster["avg_similarity"]
        ) >= MIN_CLUSTER_REPRESENTATIVE_SIMILARITY
    )


def passes_coverage_gates(cluster: dict) -> bool:
    """True when a cluster is strong enough to be a display candidate."""
    return _meets_coverage_gates(cluster)


def selected_cluster_indices(clusters: list[dict]) -> list[int]:
    """Return indices of the clusters to display, most similar first.

    A cluster qualifies once it passes the coverage gates. Among those we
    keep the TOP_CLUSTER_COUNT most similar, plus any remaining cluster at
    or above MIN_CLUSTER_SIMILARITY, so strong stories always surface. The
    result is capped at MAX_CLUSTER_COUNT, keeping the most similar.
    """
    candidates = [
        index
        for index, cluster in enumerate(clusters)
        if _meets_coverage_gates(cluster)
    ]
    ranked = sorted(
        candidates,
        key=lambda index: clusters[index]["avg_similarity"],
        reverse=True,
    )
    selected = set(ranked[:TOP_CLUSTER_COUNT])
    selected.update(
        index
        for index in candidates
        if clusters[index]["avg_similarity"] >= MIN_CLUSTER_SIMILARITY
    )
    return sorted(
        selected,
        key=lambda index: (-clusters[index]["avg_similarity"], index),
    )[:MAX_CLUSTER_COUNT]
