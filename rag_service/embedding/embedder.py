"""
Embedder for the IRAG pipeline.
Uses BAAI/bge-large-zh-v1.5 (768-dim) via sentence-transformers.
Must match the model used during indexing.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        print(f"[Embedder] Loading {model_name} on cpu...")
        self.model = SentenceTransformer(model_name, device="cpu")
        self.dim = self.model.get_sentence_embedding_dimension()
        self.text_dim = self.dim
        self.table_dim = self.dim
        print(f"[Embedder] Ready. dim={self.dim}")

    def embed_text(self, texts: list) -> list:
        """
        Returns list of np.ndarray (one per text), shape (dim,), float32, L2-normalised.
        Matches upstream indexer.py which calls embedder.embed_text([text])[0].
        """
        if not texts:
            return []
        texts = [str(t).strip() if t else "" for t in texts]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [e.astype(np.float32) for e in embeddings]

    def embed_table(self, header: list, rows: list) -> np.ndarray:
        """
        Table-level embedding: flatten header+rows into text, then embed.
        Matches upstream indexer.py: embed_table(sub_table["header"][:10], sub_table["rows"][:5])
        """
        table_text = " | ".join(str(h) for h in header) + "\n"
        for row in rows:
            table_text += " | ".join(str(c) for c in row) + "\n"
        return self.embed_text([table_text])[0]

    def embed_query_table(self, query: str) -> np.ndarray:
        return self.embed_text([query])[0]
