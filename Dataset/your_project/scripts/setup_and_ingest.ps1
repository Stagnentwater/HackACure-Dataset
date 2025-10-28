# Create venv, install deps, ingest PDFs
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\scripts\ingest.py
