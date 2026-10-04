import json
import numpy as np
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

INPUT_FILE = "headlines.json"
OUTPUT_FILE = "embeddings.npz"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    # 1. Load the headlines we collected in Task 2.
    with open(INPUT_FILE, encoding="utf-8") as f:
        articles = json.load(f)

    if not articles:
        print("No headlines found. Run ingest.py first.")
        return

    headlines = [a["title"] for a in articles]
    print(f"Loaded {len(headlines)} headlines.")

    # 2. Load the model. First run downloads ~470 MB from Hugging Face.
    print(f"Loading model {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

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
        OUTPUT_FILE,
        embeddings=embeddings,
        titles=np.array(headlines, dtype=object),
        sources=np.array([a["source"] for a in articles], dtype=object),
        countries=np.array([a["country"] for a in articles], dtype=object),
        links=np.array([a["link"] for a in articles], dtype=object),
    )
    print(f"Saved to {OUTPUT_FILE}")

    def most_similar(query, top_k=5):
        q = model.encode([query], normalize_embeddings=True)[0]
        sims = embeddings @ q                      # dot product = cosine sim
        order = np.argsort(-sims)[:top_k]
        for i in order:
            print(f"  {sims[i]:.3f}  {headlines[i]}")

    print("\nSimilar to 'ukraine war':")
    most_similar("ukraine war")

if __name__ == "__main__":
    main()