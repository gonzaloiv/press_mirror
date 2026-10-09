import argparse
import json
import sys
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import HF_TOKEN
from shared.editions import DEFAULT_EDITION_KEY, EDITIONS, get_edition

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def latest_headlines_file(history_dir: Path) -> Path:
    files = sorted(history_dir.glob("headlines_????-??-??.json"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No dated headline files found in {history_dir}")
    return files[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed each edition's headlines into a shared artifact."
    )
    parser.add_argument(
        "--edition",
        default=DEFAULT_EDITION_KEY,
        choices=sorted(EDITIONS),
        help="Edition to embed (default: %(default)s).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    edition = get_edition(args.edition)

    # 1. Load the headlines we collected for this edition.
    input_file = latest_headlines_file(edition.history_dir)
    with input_file.open(encoding="utf-8") as f:
        articles = json.load(f)

    if not articles:
        raise ValueError(f"No headlines found in {input_file}. Run ingest.py first.")

    output_file = edition.embeddings_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    headlines = [a["title"] for a in articles]
    print(f"Loaded {len(headlines)} headlines for edition '{edition.key}'.")

    # 2. Load the model. First run downloads ~470 MB from Hugging Face.
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, token=HF_TOKEN)

    # 3. Encode. This is the entire "AI" step.
    print("Encoding...")
    embeddings = model.encode(
        headlines,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,   # see note below — important
    )

    print(f"Embeddings shape: {embeddings.shape}")

    # 4. Save embeddings + metadata together so we can always line them up.
    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        titles=np.array(headlines, dtype=object),
        sources=np.array([a["source"] for a in articles], dtype=object),
        countries=np.array([a["country"] for a in articles], dtype=object),
        links=np.array([a["link"] for a in articles], dtype=object),
        fetched_at=np.array([a.get("fetched_at") for a in articles], dtype=object),
    )
    print(f"Saved to {output_file}")

    def most_similar(query, top_k=5):
        q = model.encode([query], normalize_embeddings=True)[0]
        sims = embeddings @ q                      # dot product = cosine sim
        order = np.argsort(-sims)[:top_k]
        for i in order:
            print(f"  {sims[i]:.3f}  {headlines[i]}")

if __name__ == "__main__":
    main()