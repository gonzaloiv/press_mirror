import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, HF_TOKEN
from shared.snapshots import latest_headlines_file

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_DIR / "data" / "embeddings.npz"


def main():
    input_file = latest_headlines_file()
    with input_file.open(encoding="utf-8") as f:
        articles = json.load(f)

    if not articles:
        raise ValueError(f"No headlines found in {input_file}. Run ingest.py first.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    headlines = [a["title"] for a in articles]
    print(f"Loaded {len(headlines)} headlines from {input_file}.")

    # First run downloads ~470 MB from Hugging Face.
    print(f"Loading model {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, token=HF_TOKEN)

    print("Encoding...")
    embeddings = model.encode(
        headlines,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # dot products then equal cosine similarity
    )

    print(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings + metadata together so they always line up.
    np.savez_compressed(
        OUTPUT_FILE,
        snapshot_date=np.array(input_file.parent.name),
        embeddings=embeddings,
        titles=np.array(headlines, dtype=object),
        sources=np.array([a["source"] for a in articles], dtype=object),
        countries=np.array([a["country"] for a in articles], dtype=object),
        links=np.array([a["link"] for a in articles], dtype=object),
        fetched_at=np.array([a.get("fetched_at") for a in articles], dtype=object),
    )
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
