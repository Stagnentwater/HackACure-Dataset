from __future__ import annotations
from typing import List
from . import config
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken


def _token_length(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def chunk_text(text: str) -> List[str]:
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        separators=config.SEPARATOR_PRIORITY,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=_token_length,
        is_separator_regex=False,
    )
    return splitter.split_text(text)
