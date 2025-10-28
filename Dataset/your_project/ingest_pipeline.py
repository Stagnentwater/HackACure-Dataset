"""ingest_pipeline.py

Orchestrates PDF ingestion: extraction -> chunking -> embedding -> store.
This is a top-level pipeline skeleton to wire components together.
"""
from pathlib import Path
from pdf_processor import extract_text_from_pdf
from chunking_strategy import chunk_text
from vector_store import VectorStore


def ingest_folder(folder_path: str):
    folder = Path(folder_path)
    vs = VectorStore()
    for pdf in folder.glob("*.pdf"):
        text = extract_text_from_pdf(str(pdf))
        chunks = chunk_text(text)
        # TODO: embed chunks and add to vector store
        for i, chunk in enumerate(chunks):
            vid = f"{pdf.stem}_chunk_{i}"
            vs.add(vid, [0.0], {"source": str(pdf), "chunk_index": i})
    return vs


if __name__ == "__main__":
    vs = ingest_folder("medical_pdfs")
    print(f"Ingested {len(vs._store)} items")
