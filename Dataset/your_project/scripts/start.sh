#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Render working dir for this service will be rootDir (Dataset/your_project)
# Ensure defaults if not provided
export PDF_DIR="${PDF_DIR:-medical_pdfs}"
export CHROMA_DIR="${CHROMA_DIR:-data/chroma}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-100}"

# Optionally hydrate PDFs from a remote zip to avoid committing large files
if [ ! -d "$PDF_DIR" ] || [ -z "$(ls -A "$PDF_DIR" 2>/dev/null || true)" ]; then
  if [ -n "${REMOTE_PDFS_ZIP_URL:-}" ]; then
    echo "PDFs missing -> downloading from REMOTE_PDFS_ZIP_URL..."
    python - <<'PY'
import os, io, zipfile, urllib.request
pdf_dir = os.environ.get('PDF_DIR', 'medical_pdfs')
url = os.environ['REMOTE_PDFS_ZIP_URL']
os.makedirs(pdf_dir, exist_ok=True)
print(f"Downloading PDFs from {url} ...", flush=True)
with urllib.request.urlopen(url) as r:
    data = r.read()
zf = zipfile.ZipFile(io.BytesIO(data))
zf.extractall(pdf_dir)
print(f"Extracted {len(zf.namelist())} entries into {pdf_dir}", flush=True)
PY
  else
    echo "PDFs missing and REMOTE_PDFS_ZIP_URL not set -> proceeding without ingestion assets."
  fi
fi

if [ ! -d "$CHROMA_DIR" ] || [ -z "$(ls -A "$CHROMA_DIR" 2>/dev/null || true)" ]; then
  echo "Chroma store empty -> running ingestion..."
  python -m scripts.ingest
else
  echo "Chroma store present -> skipping ingestion."
fi

HOST="0.0.0.0"
PORT="${PORT:-8000}"
exec uvicorn app.api:app --host "$HOST" --port "$PORT"
