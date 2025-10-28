#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export PDF_DIR="${PDF_DIR:-medical_pdfs}"
export CHROMA_DIR="${CHROMA_DIR:-data/chroma}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-100}"

mkdir -p "$PDF_DIR" "$CHROMA_DIR"

download_pdfs() {
  local url="${REMOTE_PDFS_URL:-${REMOTE_PDFS_ZIP_URL:-}}"
  if [ -z "${url:-}" ]; then
    echo "REMOTE_PDFS_URL not set; skipping download."
    return
  fi

  echo "Downloading PDFs from: $url"
  if echo "$url" | grep -qi "drive.google.com"; then
    # Google Drive link (file or folder). gdown handles confirmation tokens.
    python -m pip install --no-input --disable-pip-version-check gdown==5.2.0 >/dev/null 2>&1 || true
    if echo "$url" | grep -qi "/folders/"; then
      gdown --fuzzy --folder "$url" -O "$PDF_DIR"
    else
      # Could be a ZIP or single PDF
      TMP="/tmp/pdfs_download"
      rm -rf "$TMP"; mkdir -p "$TMP"
      gdown --fuzzy "$url" -O "$TMP"
      file="$(ls -1 "$TMP" | head -n1 || true)"
      if [ -z "$file" ]; then
        echo "No file downloaded from Drive."
        return
      fi
      if echo "$file" | grep -qi "\.zip$"; then
        python - <<'PY'
import sys, zipfile, os
src_dir="/tmp/pdfs_download"
pdf_dir=os.environ.get("PDF_DIR","medical_pdfs")
for name in os.listdir(src_dir):
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(os.path.join(src_dir, name)) as z:
            z.extractall(pdf_dir)
print("Extracted ZIP to", pdf_dir)
PY
      else
        mv "$TMP"/* "$PDF_DIR"/
      fi
    fi
  else
    # Generic HTTP(S) URL; try to download a ZIP and extract or copy PDFs
    TMP="/tmp/pdfs.zip"
    curl -L "$url" -o "$TMP"
    if file "$TMP" | grep -qi zip; then
      python - <<'PY'
import sys, zipfile, os
src="/tmp/pdfs.zip"
pdf_dir=os.environ.get("PDF_DIR","medical_pdfs")
with zipfile.ZipFile(src) as z: z.extractall(pdf_dir)
print("Extracted ZIP to", pdf_dir)
PY
    else
      mv "$TMP" "$PDF_DIR"/
    fi
  fi
  echo "PDF download step complete."
}

# If PDF_DIR empty, try to hydrate from REMOTE_PDFS_URL
if [ -z "$(ls -A "$PDF_DIR" 2>/dev/null || true)" ]; then
  download_pdfs
fi

# Ingest once if vector store is empty
if [ ! -d "$CHROMA_DIR" ] || [ -z "$(ls -A "$CHROMA_DIR" 2>/dev/null || true)" ]; then
  echo "Chroma store empty -> running ingestion..."
  python -m scripts.ingest
else
  echo "Chroma store present -> skipping ingestion."
fi

HOST="0.0.0.0"
PORT="${PORT:-8000}"
exec uvicorn app.api:app --host "$HOST" --port "$PORT"
