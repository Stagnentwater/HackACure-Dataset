from __future__ import annotations
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import QueryRequest, QueryResponse
from . import config
from .logging_config import configure_logging
from .rag_pipeline import answer_query

logger = configure_logging()
app = FastAPI(title="Medical RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, status_code=200)
async def query_endpoint(req: QueryRequest):
    if not req.query or not isinstance(req.query, str):
        raise HTTPException(status_code=400, detail="Invalid 'query' field")

    top_k = req.top_k if req.top_k is not None else config.DEFAULT_TOP_K
    if not isinstance(top_k, int) or top_k <= 0:
        raise HTTPException(status_code=400, detail="Invalid 'top_k' field")
    if top_k > config.MAX_TOP_K:
        top_k = config.MAX_TOP_K

    try:
        # enforce timeout
        async def _run():
            # run blocking LLM pipeline in thread to avoid blocking loop
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, answer_query, req.query, top_k, (req.mode or "default"))

        answer, contexts = await asyncio.wait_for(_run(), timeout=config.API_REQUEST_TIMEOUT_SECONDS)
        return QueryResponse(answer=answer, contexts=contexts)

    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail="Request timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


# Entrypoint for uvicorn: uvicorn app.api:app --host 0.0.0.0 --port 8000
