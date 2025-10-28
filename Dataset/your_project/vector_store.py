"""vector_store.py

Lightweight vector store interface (stub). Integrate with FAISS, Milvus, or Pinecone later.
"""

from typing import List, Tuple


class VectorStore:
    def __init__(self):
        self._store = []  # list of tuples (id, vector, metadata)

    def add(self, id: str, vector: List[float], metadata: dict):
        self._store.append((id, vector, metadata))

    def query(self, vector: List[float], top_k: int = 5) -> List[Tuple[str, float, dict]]:
        # TODO: implement real similarity search; return dummy results for now
        return [(item[0], 0.0, item[2]) for item in self._store[:top_k]]


if __name__ == "__main__":
    vs = VectorStore()
    vs.add("doc1", [0.1, 0.2], {"title": "Anatomy"})
    print(vs.query([0.1, 0.2]))
