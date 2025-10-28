from __future__ import annotations
from typing import List
import os
from . import config
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = config.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Please set it in the environment or .env file.")
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=api_key,
        task_type="retrieval_document",
    )


def embed_texts(texts: List[str]) -> List[List[float]]:
    emb = get_embeddings()
    return emb.embed_documents(texts)
