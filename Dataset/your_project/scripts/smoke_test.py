import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag_pipeline import answer_query


def run(q: str, k: int = 8, mode: str = "extractive"):
    ans, ctxs = answer_query(q, k, mode)
    print("Q:", q)
    print("Contexts:", len(ctxs))
    print("Answer:\n", ans)
    print("-" * 40)


if __name__ == "__main__":
    run("According to the provided context, list 5 common symptoms of acute myocardial infarction.")
    run("What is nephrology?", k=6, mode="default")
