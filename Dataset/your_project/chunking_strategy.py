"""chunking_strategy.py

Defines a smart chunking strategy for long medical documents.
"""

from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks with overlap (simple implementation).

    Args:
        text: the input text
        chunk_size: target chunk size in characters
        overlap: number of overlapping characters between chunks

    Returns:
        list of text chunks
    """
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = max(end - overlap, end)
    return chunks


if __name__ == "__main__":
    sample = "This is a sample medical text. " * 100
    print(len(chunk_text(sample, chunk_size=200, overlap=50)))
