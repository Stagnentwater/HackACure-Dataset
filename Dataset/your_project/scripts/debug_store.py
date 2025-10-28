import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.vectorstore import VectorStore


def main():
    vs = VectorStore()
    # Access underlying Chroma collection (private API; best-effort for diagnostics)
    try:
        coll = getattr(vs._vs, "_collection", None)
        if coll is None:
            print("Unable to access Chroma underlying collection; update script if API changes.")
            return
        data = coll.get(include=["metadatas"], limit=0)  # limit=0 => fetch all
        metas = data.get("metadatas", []) or []
        counts = {}
        for m in metas:
            src = (m or {}).get("source", "UNKNOWN")
            counts[src] = counts.get(src, 0) + 1
        total = sum(counts.values())
        print(f"Total chunks: {total}")
        for k in sorted(counts):
            print(f"{k}: {counts[k]}")
    except Exception as e:
        print(f"Failed to read collection: {e}")


if __name__ == "__main__":
    main()
