from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from .embeddings import get_embeddings
from . import config


class VectorStore:
    def __init__(self, persist_dir: Path | str | None = None):
        self.persist_dir = Path(persist_dir or config.CHROMA_DIR)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = get_embeddings()
        self._vs = Chroma(
            collection_name="medical-docs",
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] | None = None, ids: List[str] | None = None):
        docs = [Document(page_content=t, metadata=(metadatas[i] if metadatas else {})) for i, t in enumerate(texts)]
        self._vs.add_documents(docs, ids=ids)
        self._vs.persist()

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        return self._vs.similarity_search(query, k=k)

    def delete(self):
        self._vs.delete_collection()
