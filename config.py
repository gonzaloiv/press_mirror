import os


HF_TOKEN = os.environ.get("HF_TOKEN")

# Shared by embed.py (snapshot embeddings) and labels.py (term concepts).
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
