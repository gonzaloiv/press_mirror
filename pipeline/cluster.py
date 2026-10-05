import json
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from unicodedata import name as unicode_name

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

from shared.country_codes import to_iso

INPUT_FILE = PROJECT_DIR / "data" / "embeddings.npz"
HISTORY_DIR = PROJECT_DIR / "history"

SIMILARITY_THRESHOLD = 0.75    # tune this — see notes below
MERGE_SIMILARITY_THRESHOLD = 0.72
MAX_MERGE_PASSES = 3
DEDUPLICATE_WITHIN_CLUSTERS = True
MERGE_RELATED_CLUSTERS = True
LABEL_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LABEL_MAX_WORDS = 12
LABEL_MAX_NEW_TOKENS = 48


@dataclass
class ClusterMembers:
    indices: list[int]
    dropped_sources: dict[str, int] = field(default_factory=dict)
    merged: bool = False


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


def load_label_model():
    """Load the local text-generation model used to summarize clusters."""
    print(f"Loading label model {LABEL_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(LABEL_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LABEL_MODEL_NAME)
    model.eval()
    return tokenizer, model


def generate_topic_label(titles, tokenizer, model):
    """Generate an English topic label, retrying if it includes another script."""
    headlines = "\n".join(f"- {title}" for title in titles)
    system_message = (
        "Write one concise topic label entirely in English, using at most "
        f"{LABEL_MAX_WORDS} words. The headlines may be in any language: "
        "understand and translate their shared topic into natural English. "
        "Never copy a non-English phrase into the label. Use a clear, "
        "natural noun phrase, not a sentence. Name only the shared subject; "
        "do not add facts, entities, numbers, dates, results, predictions, "
        "causes, or relationships that are not supported by the headlines. "
        "Output only the English label."
    )

    def generate(system_content):
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": f"Headlines:\n{headlines}"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=LABEL_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_tokens = output[0, inputs["input_ids"].shape[-1]:]
        return (
            tokenizer.decode(new_tokens, skip_special_tokens=True)
            .strip()
            .strip('"')
        )

    def has_non_latin_letters(text):
        """Detect script leakage without rejecting accented English names."""
        return any(
            character.isalpha()
            and not unicode_name(character, "").startswith("LATIN ")
            for character in text
        )

    label = generate(system_message)
    if has_non_latin_letters(label):
        label = generate(
            system_message
            + " Your previous draft contained non-English characters. Translate "
            "the entire topic into natural English, using only the Latin alphabet."
        )
        if has_non_latin_letters(label):
            label = generate(
                system_message
                + f" Translate this draft into English: {label}. "
                "Use only the Latin alphabet."
            )
        if has_non_latin_letters(label):
            first_non_latin = next(
                index
                for index, character in enumerate(label)
                if character.isalpha()
                and not unicode_name(character, "").startswith("LATIN ")
            )
            label = label[:first_non_latin].rstrip(" ,;:-")
            print(
                "Warning: removed a non-Latin suffix from a generated label.",
                file=sys.stderr,
            )
    if not label:
        raise RuntimeError("The label model returned an empty cluster summary.")
    source_numbers = set(re.findall(r"\b\d+\b", headlines))
    label = re.sub(
        r"\b\d+\b",
        lambda match: match.group() if match.group() in source_numbers else "",
        label,
    )
    label = " ".join(label.split())
    if not label:
        raise RuntimeError("The label model returned no supported cluster summary.")
    return label


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

    result = []
    label_clusters = [
        cluster for cluster in cluster_groups if len(cluster.indices) >= 2
    ]
    print(
        f"Selecting labels for {len(label_clusters)} clusters.",
        flush=True,
    )
    for label_number, cluster in enumerate(label_clusters, start=1):
        indices = cluster.indices

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

        print(f"Generating label {label_number}/{len(label_clusters)}...", flush=True)
        cluster_label = labeler([titles[i] for i in indices])

        cluster_result = {
            "label": cluster_label,
            "size": len(indices),
            "avg_similarity": round(float(avg_sim), 3),
            "articles": articles,
        }
        if cluster.dropped_sources:
            cluster_result["dropped_sources"] = cluster.dropped_sources
        result.append(cluster_result)

    # Prefer stories covered in more countries; use article count only for ties.
    result.sort(
        key=lambda c: (
            len({to_iso(article["country"]) for article in c["articles"]}),
            c["size"],
        ),
        reverse=True,
    )
    return result


def main():
    embeddings, titles, sources, countries, links = load_data()
    print(f"Loaded {len(titles)} headlines.")

    labels = cluster(embeddings)
    print(f"Found {len(set(labels))} raw clusters.")

    tokenizer, model = load_label_model()
    labeler = lambda cluster_titles: generate_topic_label(
        cluster_titles, tokenizer, model
    )
    clusters = build_clusters(
        labels,
        titles,
        sources,
        countries,
        links,
        embeddings,
        labeler,
    )
    print(f"Kept {len(clusters)} clusters with 2+ articles.")

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    output_file = HISTORY_DIR / f"clusters_{date.today().isoformat()}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"Written to {output_file}")


if __name__ == "__main__":
    main()