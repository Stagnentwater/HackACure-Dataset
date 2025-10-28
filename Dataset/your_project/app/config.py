import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Google Gemini
GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

# Chunking
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))  # tokens
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))  # tokens
SEPARATOR_PRIORITY: list[str] = [
    "\n\n\n",
    "\n\n",
    "\n",
    ". ",
    ", ",
    " ",
    "",
]

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = Path(os.getenv("PDF_DIR", PROJECT_ROOT / "medical_pdfs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "data" / "chroma"))

# API
API_REQUEST_TIMEOUT_SECONDS: int = int(os.getenv("API_REQUEST_TIMEOUT_SECONDS", "60"))
MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", "10"))
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))

# Performance / batching
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
RATE_LIMIT_QPM: int = int(os.getenv("RATE_LIMIT_QPM", "15"))

# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
