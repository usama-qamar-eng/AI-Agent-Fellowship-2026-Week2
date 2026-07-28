import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from config import COLLECTION_NAME, EMBEDDING_MODEL, PERSIST_DIR

class Store:
    def __init__(self, persist_dir: str = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
        # persist_dir/collection_name are overridable (not just read from config) so the
        # experiments/ scripts can spin up a throwaway store pointed at a temp directory
        # instead of writing into the app's real production collection.
        client = chromadb.PersistentClient(path=persist_dir)
        self.collection = client.get_or_create_collection(
            collection_name,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL),
        )

    def add(self, doc_name: str, chunks: list[dict]) -> None:
        ids = [f"{doc_name}::{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": doc_name, "page": c["page"], "chunk_id": i} for i, c in enumerate(chunks)],
        )

    def query(self, question: str, top_k: int, doc_filter: str | None = None) -> list[dict]:
        where = {"source": doc_filter} if doc_filter else None
        res = self.collection.query(query_texts=[question], n_results=top_k, where=where)
        return [
            {"text": doc, **meta}
            for doc, meta in zip(res["documents"][0], res["metadatas"][0])
        ]

    def hybrid_query(self, question: str, top_k: int, doc_filter: str | None = None) -> list[dict]:
        """Semantic search over-fetches candidates, then re-ranks by blending
        semantic rank with keyword overlap — a lightweight hybrid search with
        no extra dependency (no BM25 library needed for this scale)."""
        candidates = self.query(question, top_k=top_k * 3, doc_filter=doc_filter)
        terms = [t for t in question.lower().split() if len(t) > 2]

        def score(rank: int, chunk: dict) -> float:
            semantic = 1.0 / (rank + 1)
            keyword = sum(t in chunk["text"].lower() for t in terms) / max(len(terms), 1)
            return 0.7 * semantic + 0.3 * keyword

        ranked = sorted(enumerate(candidates), key=lambda pair: -score(*pair))
        return [chunk for _, chunk in ranked[:top_k]]

    def list_documents(self) -> dict[str, int]:
        """Returns {doc_name: chunk_count} for every document currently stored."""
        rows = self.collection.get(include=["metadatas"])["metadatas"]
        counts: dict[str, int] = {}
        for m in rows:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return counts

    def delete_document(self, doc_name: str) -> None:
        self.collection.delete(where={"source": doc_name})
