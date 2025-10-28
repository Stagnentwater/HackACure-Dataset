from typing import List, Optional
from pydantic import BaseModel, Field, validator


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query string")
    top_k: Optional[int] = Field(None, description="Number of contexts to retrieve")
    mode: Optional[str] = Field(
        None,
        description="Answering mode: 'default' (prompted reasoning with fallback) or 'extractive' (direct extraction first)",
    )

    @validator("query")
    def non_empty_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must be a non-empty string")
        return v.strip()

    @validator("top_k")
    def valid_top_k(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if v <= 0:
            raise ValueError("top_k must be a positive integer")
        return v

    @validator("mode")
    def valid_mode(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"default", "extractive"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {sorted(allowed)}")
        return v


class QueryResponse(BaseModel):
    answer: str  # Concise, correct answer
    contexts: List[str]  # Top-k relevant text snippets as plain strings
