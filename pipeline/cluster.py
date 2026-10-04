import json
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer

from country_codes import to_iso

INPUT_FILE = PROJECT_DIR / "embeddings.npz"
HISTORY_DIR = PROJECT_DIR / "history"

SIMILARITY_THRESHOLD = 0.75    # tune this — see notes below
LABEL_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LABEL_MAX_WORDS = 8
LABEL_MAX_NEW_TOKENS = 48


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


def load_label_model():
    """Load the local text-generation model used to summarize clusters."""
    print(f"Loading label model {LABEL_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(LABEL_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(LABEL_MODEL_NAME)
    model.eval()
    return tokenizer, model


def generate_topic_label(titles, tokenizer, model):
    """Generate a concise English summary from the headlines in one cluster."""
    headlines = "\n".join(f"- {title}" for title in titles)
    messages = [
        {
            "role": "system",
            "content": (
                "Create an English topic label as a noun phrase, not a sentence. "
                "Use at most "
                f"{LABEL_MAX_WORDS} words. Name only the shared subject, not what "
                "happened to it. Use no dates, results, predictions, causes, or "
                "relationships. Do not introduce entities or facts absent from the "
                "headlines. For headlines in other languages, translate the shared "
                "topic into English. Output only the label."
            ),
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
    label = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().strip('"')
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
    words = label.split()
    if len(words) > LABEL_MAX_WORDS:
        label = " ".join(words[:LABEL_MAX_WORDS - 1]) + "..."
    return label


def build_clusters(labels, titles, sources, countries, links, embeddings, labeler):
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
            "label": labeler([titles[i] for i in indices]),
            "size": len(indices),
            "avg_similarity": round(float(avg_sim), 3),
            "articles": articles,
        })

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
    labeler = lambda cluster_titles: generate_topic_label(cluster_titles, tokenizer, model)
    clusters = build_clusters(
        labels, titles, sources, countries, links, embeddings, labeler
    )
    print(f"Kept {len(clusters)} clusters with 2+ articles.")

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    output_file = HISTORY_DIR / f"clusters_{date.today().isoformat()}.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    print(f"Written to {output_file}")


if __name__ == "__main__":
    main()