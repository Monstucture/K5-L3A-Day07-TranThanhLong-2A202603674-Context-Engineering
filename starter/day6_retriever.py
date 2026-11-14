import os, sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.environ["DAY6_STARTER"])  # reuse Day 06 code as-is
from src.embeddings import MyEmbedder
from src.store import PgVectorStore


class Day6Retriever:
    def __init__(self, strategy: str = "B"):
        self.strategy = strategy
        self.embedder = MyEmbedder(os.environ["EMBEDDING_MODEL"])
        self.store = PgVectorStore(os.environ["DATABASE_URL"], self.embedder.dimensions)

    def search(self, query, *, filters=None, top_k=3):
        rows = self.store.search(self.strategy, self.embedder.embed_query(query), top_k, filters)
        # Map Day 06 columns to the Day 07 chunk format (data/README.md)
        return [{"id": r["chunk_id"], "source_id": r["document_id"], "source": r["source"],
                 "page": r["page"], "text": r["text"], "score": float(r["score"]),
                 "metadata": {k: r[k] for k in ("page_end", "chapter", "section", "section_title", "content_type")}}
                for r in rows]
