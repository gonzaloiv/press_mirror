import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from shared.country_codes import to_iso
from shared.cluster_selection import passes_coverage_gates, selected_cluster_indices
from shared.snapshots import headlines_file, latest_headlines_file, snapshot_dir
from pipeline.clustering.labels import build_cluster_label, extract_common_terms
from pipeline.clustering.llm_label import (
    english_headline,
    generate_llm_label_reviewed,
    translate_shared_terms,
)

INPUT_FILE = PROJECT_DIR / "data" / "embeddings.npz"

GOOD_LABEL_SOURCES = frozenset({"llm", "headline"})

SIMILARITY_THRESHOLD = 0.75    # tune this — see notes below
MERGE_SIMILARITY_THRESHOLD = 0.72
MAX_MERGE_PASSES = 3
DEDUPLICATE_WITHIN_CLUSTERS = True
MERGE_RELATED_CLUSTERS = True


@dataclass
class ClusterMembers:
    indices: list[int]
    dropped_sources: dict[str, int] = field(default_factory=dict)
    merged: bool = False


def load_data():
    data = np.load(INPUT_FILE, allow_pickle=True)
    titles = list(data["titles"])
    snapshot_date = (
        str(data["snapshot_date"].item())
        if "snapshot_date" in data.files
        else latest_headlines_file().parent.name
    )
    source_file = headlines_file(snapshot_date)
    if not source_file.is_file():
        raise FileNotFoundError(
            f"Embeddings refer to {snapshot_date}, but its headlines file is "
            f"missing at {source_file}. Run ingest.py and embed.py first."
        )
    return (
        snapshot_date,
        data["embeddings"],
        titles,
        list(data["sources"]),
        list(data["countries"]),
        list(data["links"]),
        (
            list(data["fetched_at"])
            if "fetched_at" in data.files
            else [None] * len(titles)
        ),
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


def deduplicate_cluster(
    indices: list[int],
    sources: list[str],
    embeddings: np.ndarray,
) -> ClusterMembers:
    """Keep the article nearest its cluster centroid for each source."""
    if not indices:
        return ClusterMembers(indices=[])

    centroid = embeddings[indices].mean(axis=0)
    similarities = cosine_similarity(
        embeddings[indices],
        centroid.reshape(1, -1),
    ).ravel()
    best_by_source: dict[str, tuple[float, int]] = {}
    source_counts: dict[str, int] = {}
    for index, similarity in zip(indices, similarities):
        source = sources[index]
        source_counts[source] = source_counts.get(source, 0) + 1
        current = best_by_source.get(source)
        if current is None or similarity > current[0]:
            best_by_source[source] = (float(similarity), index)

    kept_indices = sorted(entry[1] for entry in best_by_source.values())
    dropped_sources = {
        source: count - 1
        for source, count in source_counts.items()
        if count > 1
    }
    return ClusterMembers(
        indices=kept_indices,
        dropped_sources=dropped_sources,
    )


def deduplicate_clusters(
    clusters: list[list[int]],
    sources: list[str],
    embeddings: np.ndarray,
) -> list[ClusterMembers]:
    """Deduplicate each primary cluster independently by source."""
    return [
        deduplicate_cluster(indices, sources, embeddings)
        for indices in clusters
        if indices
    ]


def merge_clusters(
    clusters: list[ClusterMembers],
    sources: list[str],
    embeddings: np.ndarray,
    threshold: float = MERGE_SIMILARITY_THRESHOLD,
    max_passes: int = MAX_MERGE_PASSES,
) -> list[ClusterMembers]:
    """Merge similar cluster centroids and deduplicate merged article lists."""
    current = [cluster for cluster in clusters if cluster.indices]
    for pass_number in range(1, max_passes + 1):
        centroids = [
            embeddings[cluster.indices].mean(axis=0)
            for cluster in current
        ]
        print(
            f"Merge pass {pass_number}: comparing {len(current)} cluster centroids.",
            flush=True,
        )
        similarity_matrix = cosine_similarity(centroids)
        matching_pairs = np.argwhere(
            np.triu(similarity_matrix >= threshold, k=1)
        )
        pair_scores = [
            (float(similarity_matrix[left, right]), int(left), int(right))
            for left, right in matching_pairs
        ]

        if not pair_scores:
            break

        paired_left: dict[int, int] = {}
        paired_right: set[int] = set()
        for _, left, right in sorted(pair_scores, reverse=True):
            if left not in paired_left and left not in paired_right \
                    and right not in paired_left and right not in paired_right:
                paired_left[left] = right
                paired_right.add(right)

        next_clusters = []
        for index, cluster in enumerate(current):
            if index in paired_right:
                continue
            partner = paired_left.get(index)
            if partner is None:
                next_clusters.append(cluster)
                continue

            other = current[partner]
            combined = deduplicate_cluster(
                cluster.indices + other.indices,
                sources,
                embeddings,
            )
            for dropped_sources in (
                cluster.dropped_sources,
                other.dropped_sources,
            ):
                for source, count in dropped_sources.items():
                    combined.dropped_sources[source] = (
                        combined.dropped_sources.get(source, 0) + count
                    )
            combined.merged = True
            next_clusters.append(combined)

        print(
            f"Merge pass {pass_number}: {len(current)} -> "
            f"{len(next_clusters)} clusters."
        )
        current = next_clusters

        if pass_number == max_passes:
            final_centroids = [
                embeddings[cluster.indices].mean(axis=0)
                for cluster in current
            ]
            final_similarities = cosine_similarity(final_centroids)
            if np.any(np.triu(final_similarities >= threshold, k=1)):
                print(
                    f"Warning: similar clusters remain after {max_passes} "
                    "merge passes."
                )

    return current


def build_clusters(
    labels,
    titles,
    sources,
    countries,
    links,
    embeddings,
    labeler,
    deduplicate: bool = DEDUPLICATE_WITHIN_CLUSTERS,
    merge: bool = MERGE_RELATED_CLUSTERS,
    fetched_at=None,
):
    """Post-process primary clusters, then build output with similarity stats."""
    raw_clusters = {}
    for idx, label in enumerate(labels):
        raw_clusters.setdefault(int(label), []).append(idx)

    cluster_groups = list(raw_clusters.values())
    if deduplicate:
        cluster_groups = deduplicate_clusters(cluster_groups, sources, embeddings)
        dropped_count = sum(
            sum(cluster.dropped_sources.values())
            for cluster in cluster_groups
        )
        print(f"Dropped {dropped_count} same-source articles.")
    else:
        cluster_groups = [
            ClusterMembers(indices=indices)
            for indices in cluster_groups
            if indices
        ]
    if merge:
        cluster_groups = merge_clusters(cluster_groups, sources, embeddings)

    entries = []
    label_clusters = [
        cluster for cluster in cluster_groups if len(cluster.indices) >= 2
    ]
    print(
        f"Measuring {len(label_clusters)} clusters.",
        flush=True,
    )
    for cluster in label_clusters:
        indices = cluster.indices

        # Recompute pairwise similarity *within* this cluster for the display.
        sub = embeddings[indices]
        sim_matrix = cosine_similarity(sub)
        # Average of the off-diagonal (exclude self-similarity = 1.0).
        n = len(indices)
        avg_sim = (sim_matrix.sum() - n) / (n * n - n)

        articles = []
        for i in indices:
            article = {
                "title": titles[i],
                "source": sources[i],
                "country": countries[i],
                "link": links[i],
            }
            if fetched_at is not None and fetched_at[i] is not None:
                article["fetched_at"] = str(fetched_at[i])
            articles.append(article)

        centroid = sub.mean(axis=0)
        similarities = cosine_similarity(
            sub,
            centroid.reshape(1, -1),
        ).ravel()
        representative_position = int(np.argmax(similarities))
        min_representative_similarity = sim_matrix[
            representative_position
        ].min()
        ranked_positions = np.argsort(-similarities, kind="stable")
        cluster_titles = [
            titles[indices[position]] for position in ranked_positions
        ]

        cluster_result = {
            "label": "",
            "terms": [],
            "size": len(indices),
            "avg_similarity": round(float(avg_sim), 3),
            "min_representative_similarity": round(
                float(min_representative_similarity), 3
            ),
            "articles": articles,
        }
        if cluster.dropped_sources:
            cluster_result["dropped_sources"] = cluster.dropped_sources
        entries.append((cluster_result, cluster_titles))

    # Prefer stories covered in more countries; use article count only for ties.
    entries.sort(
        key=lambda entry: (
            len({to_iso(a["country"]) for a in entry[0]["articles"]}),
            entry[0]["size"],
        ),
        reverse=True,
    )
    result = [entry[0] for entry in entries]

    # Labels are only needed for clusters that pass the display filters.
    shown = selected_cluster_indices(result)
    print(
        f"Generating labels for {len(shown)} displayed clusters.",
        flush=True,
    )
    for label_number, index in enumerate(shown, start=1):
        cluster_result, cluster_titles = entries[index]
        print(f"Generating label {label_number}/{len(shown)}...", flush=True)
        cluster_terms = labeler(cluster_titles)
        english_terms = translate_shared_terms(cluster_terms, cluster_titles)
        label_terms = english_terms or cluster_terms
        llm_label, review_reason = generate_llm_label_reviewed(
            label_terms, cluster_titles
        )
        review_action = (
            "kept generated label" if llm_label else "using term fallback"
        )
        print(
            f"Label review {label_number}/{len(shown)}: "
            f"{review_reason}; {review_action}.",
            flush=True,
        )
        cluster_result["source_terms"] = cluster_terms
        cluster_result["terms"] = english_terms
        if llm_label:
            cluster_result["label"] = llm_label
            cluster_result["label_source"] = "llm"
        else:
            fallback = english_headline(cluster_titles)
            if fallback:
                cluster_result["label"] = fallback
                cluster_result["label_source"] = "headline"
            else:
                cluster_result["label"] = build_cluster_label(
                    english_terms or cluster_terms, cluster_titles[0]
                )
                cluster_result["label_source"] = "terms"
        cluster_result["label_review"] = {
            "status": "accepted" if llm_label else "fallback",
            "reason": review_reason,
        }

    # A display candidate without a good English name is dropped, so the site
    # only shows stories it can name cleanly.
    kept = [
        cluster
        for cluster in result
        if cluster.get("label_source") in GOOD_LABEL_SOURCES
        or not passes_coverage_gates(cluster)
    ]
    dropped = len(result) - len(kept)
    if dropped:
        print(f"Dropped {dropped} clusters without a usable label.", flush=True)
    return kept


def main():
    snapshot_date, embeddings, titles, sources, countries, links, fetched_at = load_data()
    print(f"Loaded {len(titles)} headlines.")

    labels = cluster(embeddings)
    print(f"Found {len(set(labels))} raw clusters.")

    clusters = build_clusters(
        labels,
        titles,
        sources,
        countries,
        links,
        embeddings,
        extract_common_terms,
        fetched_at=fetched_at,
    )
    print(f"Kept {len(clusters)} clusters with 2+ articles.")

    snapshot_path = snapshot_dir(snapshot_date)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    output_file = snapshot_path / "clusters.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"Written to {output_file}")


if __name__ == "__main__":
    main()