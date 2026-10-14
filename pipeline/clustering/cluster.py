import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from shared.country_codes import to_iso
from shared.cluster_selection import (
    TOP_CLUSTER_COUNT,
    passes_coverage_gates,
    selected_cluster_indices,
)
from shared.snapshots import headlines_file, latest_headlines_file, snapshot_dir
from pipeline.clustering.labels import build_cluster_label, extract_common_terms
from pipeline.clustering.llm_label import (
    clean_headline,
    english_headline,
    generate_llm_label_reviewed,
    generator_available,
    translate_shared_terms,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "embeddings.npz"


class SnapshotData(NamedTuple):
    """One snapshot's embeddings and the article fields clustered from them."""

    snapshot_date: str
    embeddings: np.ndarray
    titles: list[str]
    sources: list[str]
    countries: list[str]
    links: list[str]
    fetched_at: list

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
    return SnapshotData(
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


def load_summaries(snapshot_date: str) -> dict[str, str]:
    """Map article links to their RSS summaries for label context.

    Summaries are only used to give the label model more context, so a
    snapshot without them (or without a headlines file) simply yields none.
    """
    path = headlines_file(snapshot_date)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        headlines = json.load(f)
    summaries = {}
    for article in headlines:
        link = article.get("link")
        summary = article.get("summary")
        if link and summary:
            summaries[link] = summary
    return summaries


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

    if current:
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


def _cluster_entry(
    cluster,
    titles,
    sources,
    countries,
    links,
    embeddings,
    fetched_at,
):
    """Build one output cluster and its centroid-ranked titles for labeling."""
    indices = cluster.indices
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
    similarities = cosine_similarity(sub, centroid.reshape(1, -1)).ravel()
    representative_position = int(np.argmax(similarities))
    min_representative_similarity = sim_matrix[representative_position].min()
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
    return cluster_result, cluster_titles


def _label_cluster(cluster_result, cluster_titles, labeler, summaries=None):
    """Fill in one shown cluster's terms and label; return the label outcome."""
    summaries = summaries or {}
    context = [
        summaries[article["link"]]
        for article in cluster_result["articles"]
        if article.get("link") in summaries
    ]
    cluster_terms = labeler(cluster_titles)
    english_terms = translate_shared_terms(
        cluster_terms, cluster_titles, summaries=context
    )
    label_terms = english_terms or cluster_terms
    llm_label, review_reason = generate_llm_label_reviewed(
        label_terms, cluster_titles, summaries=context
    )
    cluster_result["source_terms"] = cluster_terms
    cluster_result["terms"] = english_terms
    if llm_label:
        cluster_result["label"] = llm_label
        cluster_result["label_source"] = "llm"
    else:
        fallback = english_headline(cluster_titles)
        if fallback:
            cluster_result["label"] = clean_headline(fallback)
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
    return llm_label, review_reason


def _ranked_gate_candidates(result) -> list[int]:
    """Gated clusters, most similar first (the app's display ranking)."""
    ranked = [
        index
        for index, cluster in enumerate(result)
        if passes_coverage_gates(cluster)
    ]
    ranked.sort(key=lambda index: (-result[index]["avg_similarity"], index))
    return ranked


def _label_in_order(
    result,
    entries,
    labeler,
    summaries,
    order: list[int],
    first_number: int,
    total: int,
):
    """Label each candidate cluster in order; return the clean-label count."""
    clean = 0
    for offset, index in enumerate(order):
        cluster_result, cluster_titles = entries[index]
        label_number = first_number + offset
        print(f"Generating label {label_number}/{total}...", flush=True)
        llm_label, review_reason = _label_cluster(
            cluster_result, cluster_titles, labeler, summaries
        )
        review_action = (
            "kept generated label" if llm_label else "using term fallback"
        )
        print(
            f"Label review {label_number}/{total}: "
            f"{review_reason}; {review_action}.",
            flush=True,
        )
        if cluster_result.get("label_source") in GOOD_LABEL_SOURCES:
            clean += 1
    return clean


def _prune_without_labels(
    result, ranked: list[int], llm_available: bool = True
):
    """Keep the daily five named: clean labels only, fallbacks dropped.

    A gated cluster with a clean name (LLM or headline) is always kept. A
    keyword-only fallback shares the defect of a rejected label, so it is
    dropped rather than shown or used to pad the daily selection. The lone
    exception is a label model that cannot run at all: then the fallbacks
    fill the remaining slots so a broken model still shows a page.
    """
    good = [
        index
        for index in ranked
        if result[index].get("label_source") in GOOD_LABEL_SOURCES
    ]
    keep = set(good)
    if not llm_available and len(good) < TOP_CLUSTER_COUNT:
        fallbacks = [
            index
            for index in ranked
            if result[index].get("label_source")
            and result[index].get("label_source") not in GOOD_LABEL_SOURCES
        ]
        keep.update(fallbacks[: TOP_CLUSTER_COUNT - len(good)])
    kept = [
        cluster
        for index, cluster in enumerate(result)
        if index in keep or not passes_coverage_gates(cluster)
    ]
    dropped = len(result) - len(kept)
    if dropped:
        print(f"Dropped {dropped} clusters without a usable label.", flush=True)
    return kept


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
    summaries=None,
    llm_available: bool = True,
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

    label_clusters = [
        cluster for cluster in cluster_groups if len(cluster.indices) >= 2
    ]
    print(
        f"Measuring {len(label_clusters)} clusters.",
        flush=True,
    )
    entries = [
        _cluster_entry(
            cluster, titles, sources, countries, links, embeddings, fetched_at
        )
        for cluster in label_clusters
    ]

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
    ranked = _ranked_gate_candidates(result)
    shown = selected_cluster_indices(result)
    print(
        f"Generating labels for {len(shown)} displayed clusters.",
        flush=True,
    )
    good_labels = _label_in_order(
        result, entries, labeler, summaries, shown, 1, len(shown)
    )
    if good_labels < TOP_CLUSTER_COUNT:
        # The daily grid wants five named stories, so label deeper gated
        # candidates in similarity order until five carry a clean label.
        deeper = [index for index in ranked if index not in set(shown)]
        if deeper:
            total = len(shown) + len(deeper)
            print(
                f"{good_labels}/{len(shown)} clean labels; labeling "
                f"{len(deeper)} more candidates to fill the daily selection.",
                flush=True,
            )
            good_labels += _label_in_order(
                result,
                entries,
                labeler,
                summaries,
                deeper,
                len(shown) + 1,
                total,
            )

    if not llm_available:
        # The model could not run at all, so every label is a fallback.
        print(
            "Label model unavailable; keeping fallback-labeled clusters.",
            flush=True,
        )
    return _prune_without_labels(result, ranked, llm_available)


def main():
    data = load_data()
    print(f"Loaded {len(data.titles)} headlines.")

    labels = cluster(data.embeddings)
    print(f"Found {len(set(labels))} raw clusters.")

    llm_available = generator_available()
    if not llm_available:
        print(
            "Label model unavailable; labels will fall back to shared terms.",
            flush=True,
        )

    summaries = load_summaries(data.snapshot_date)
    if summaries:
        print(f"Loaded {len(summaries)} RSS summaries for label context.")

    clusters = build_clusters(
        labels,
        data.titles,
        data.sources,
        data.countries,
        data.links,
        data.embeddings,
        extract_common_terms,
        fetched_at=data.fetched_at,
        summaries=summaries,
        llm_available=llm_available,
    )
    print(f"Kept {len(clusters)} clusters with 2+ articles.")

    snapshot_path = snapshot_dir(data.snapshot_date)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    output_file = snapshot_path / "clusters.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"Written to {output_file}")


if __name__ == "__main__":
    main()