# Medical RAG API (Gemini + ChromaDB)

Production-ready Retrieval-Augmented Generation (RAG) API for medical PDFs using Google Gemini, LangChain chunking, and a local persistent ChromaDB.

## Features
- Cost-effective: Free embeddings (`models/text-embedding-004`) and low-cost LLM (`gemini-2.5-flash`)
- PDF processing with PyMuPDF (robust for medical docs)
- Token-aware chunking: 512 tokens, 100 overlap with custom separators
- Persistent local vector DB with Chroma
- FastAPI endpoint `/query` with 60s timeout and strict response model
- RAGAS evaluation script and sample test queries
- Dockerized deployment, production logging, health check

## Project Structure
```
your_project/
  app/
    __init__.py
    api.py                 # FastAPI app with /query and /health
    chunking.py            # Token-aware chunking
    config.py              # Config + env
    embeddings.py          # Gemini embeddings (LangChain wrapper)
    logging_config.py      # Logging setup
    models.py              # Pydantic request/response
    pdf_ingest.py          # PDF -> chunks -> Chroma pipeline
    rag_pipeline.py        # Retrieval + LLM answer
    vectorstore.py         # Chroma wrapper (persistent)
  data/
    chroma/                # Chroma persistence dir
  medical_pdfs/            # Place your 9 PDFs here
  scripts/
    ingest.py              # CLI to ingest PDFs
    evaluate_ragas.py      # RAGAS evaluation
    test_queries.json      # Sample test cases
  requirements.txt
  .env.example
  Dockerfile
  docker-compose.yml
  README.md
```

## Setup
1. Create API key (free): https://aistudio.google.com/app/apikey
2. Copy `.env.example` to `.env` and fill `GOOGLE_API_KEY`.
3. Put your 9 medical PDFs in `medical_pdfs/`.

### Local (Windows PowerShell)
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\scripts\ingest.py
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Docker
```powershell
# Build and run
$env:GOOGLE_API_KEY="AIza..."; docker compose up --build
```

## API
- Method: POST
- URL: /query
- Request body:
```json
{"query": "string", "top_k": 5}
```
- Response body:
```json
{"answer": "string", "contexts": ["string", "..."]}
```
- Timeout: 60 seconds
- Status codes: 200 (success), 400 (bad request), 500 (internal error)

## Evaluation (RAGAS)
```powershell
python .\scripts\evaluate_ragas.py
```
Metrics computed:
- Answer Relevancy (30%)
- Answer Correctness (30%)
- Context Relevance (25%)
- Faithfulness (15%)

## Notes
- Expected first-time ingestion: ~2–3 minutes for ~8–12k chunks.
- Chroma data persists under `data/chroma`.
- Set `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_BATCH_SIZE` etc. in `.env`.

## Troubleshooting
- 401/403 from Gemini: check `GOOGLE_API_KEY` and project quotas.
- Import errors: `pip install -r requirements.txt` inside your active venv.
- Empty answers: ensure PDFs are present and ingestion ran.
- Performance: increase `EMBEDDING_BATCH_SIZE` cautiously; keep within free tier limits.
