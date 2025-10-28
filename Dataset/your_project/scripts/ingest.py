from __future__ import annotations
import argparse
from pathlib import Path
from app.pdf_ingest import ingest_pdfs
from app.logging_config import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Chroma vector store")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Optional path to directory containing PDFs (overrides config.PDF_DIR)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing Chroma collection before ingesting",
    )
    args = parser.parse_args()

    configure_logging()
    ingest_pdfs(pdf_dir=args.pdf_dir, reset=args.reset)


if __name__ == "__main__":
    main()
