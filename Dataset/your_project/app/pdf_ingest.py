from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import logging
import fitz  # PyMuPDF
from . import config
from .chunking import chunk_text
from .vectorstore import VectorStore
from .embeddings import embed_texts

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> List[Dict]:
    """Extract text per page with metadata.

    Returns a list of dicts with keys: text, metadata
    metadata includes: source, page
    """
    results: List[Dict] = []
    try:
        with fitz.open(pdf_path) as doc:
            for i, page in enumerate(doc):
                text = page.get_text("text") or ""
                if text.strip():
                    results.append({
                        "text": text,
                        "metadata": {
                            "source": str(pdf_path),
                            "page": i + 1,
                            "total_pages": len(doc),
                        }
                    })
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
    return results


def ingest_pdfs(pdf_dir: Path | None = None, reset: bool = False):
    logging.getLogger().setLevel(logging.INFO)
    pdf_dir = Path(pdf_dir or config.PDF_DIR)
    vs = VectorStore()
    if reset:
        vs.delete()
        # Recreate the collection handle after deletion to avoid stale references
        vs = VectorStore()
    texts: List[str] = []
    metadatas: List[Dict] = []

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    for pdf in pdf_files:
        logger.info(f"Processing PDF: {pdf.name}")
        pages = extract_text_from_pdf(pdf)
        logger.info(f"\tExtracted {len(pages)} pages with text from {pdf.name}")
        file_chunk_count = 0
        for page in pages:
            chunks = chunk_text(page["text"])  # token-aware
            for idx, chunk in enumerate(chunks):
                texts.append(chunk)
                md = dict(page["metadata"])  # copy
                md.update({"chunk_index": idx})
                metadatas.append(md)
                file_chunk_count += 1
        logger.info(f"\tChunked {file_chunk_count} chunks from {pdf.name}")

    if not texts:
        logger.warning("No text chunks found to ingest.")
        return

    # Embed in batches for efficiency
    batch_size = config.EMBEDDING_BATCH_SIZE
    ids: List[str] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_mds = metadatas[i:i+batch_size]
        # pre-generate IDs
        batch_ids = [f"doc_{i+j}" for j in range(len(batch_texts))]
        # LangChain Chroma computes embeddings internally when an embedding_function is set.
        # We add documents directly to avoid duplicating embedding work.
        vs.add_texts(batch_texts, metadatas=batch_mds, ids=batch_ids)
        ids.extend(batch_ids)
        logger.info(f"Indexed {i+len(batch_texts)} / {len(texts)} chunks")

    logger.info(f"Ingestion complete. Total chunks indexed: {len(texts)}")


if __name__ == "__main__":
    ingest_pdfs()
