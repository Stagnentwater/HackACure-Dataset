#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Render working dir for this service will be rootDir (Dataset/your_project)
# Ensure defaults if not provided
export PDF_DIR="${PDF_DIR:-medical_pdfs}"
export CHROMA_DIR="${CHROMA_DIR:-data/chroma}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-100}"

if [ ! -d "$CHROMA_DIR" ] || [ -z "$(ls -A "$CHROMA_DIR" 2>/dev/null || true)" ]; then
  echo "Chroma store empty -> running ingestion..."
  python -m scripts.ingest
else
  echo "Chroma store present -> skipping ingestion."
fi

HOST="0.0.0.0"
PORT="${PORT:-8000}"
exec uvicorn app.api:app --host "$HOST" --port "$PORT"
