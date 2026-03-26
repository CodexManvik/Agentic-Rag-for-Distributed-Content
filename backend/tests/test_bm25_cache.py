from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings
from app.services import vector_store


class _FakeCollection:
    def __init__(self, docs: list[str], ids: list[str]):
        self._docs = docs
        self._ids = ids

    def get(self, include: list[str]):
        return {"ids": self._ids, "documents": self._docs}


def test_bm25_cache_refresh_consistency(monkeypatch) -> None:
    prev_cache = settings.bm25_cache_enabled
    prev_hybrid = settings.hybrid_retrieval_enabled
    settings.bm25_cache_enabled = True
    settings.hybrid_retrieval_enabled = True

    first = _FakeCollection(
        docs=["alpha retrieval context", "beta generation context"],
        ids=["id1", "id2"],
    )
    monkeypatch.setattr(vector_store, "_ensure_collection", lambda: first)
    vector_store.refresh_bm25_cache()
    first_scores = vector_store._bm25_candidates("alpha", 2)
    assert "id1" in first_scores

    second = _FakeCollection(
        docs=["gamma tool calling", "delta tracing"],
        ids=["id3", "id4"],
    )
    monkeypatch.setattr(vector_store, "_ensure_collection", lambda: second)
    vector_store.refresh_bm25_cache()
    second_scores = vector_store._bm25_candidates("gamma", 2)
    assert "id3" in second_scores

    settings.bm25_cache_enabled = prev_cache
    settings.hybrid_retrieval_enabled = prev_hybrid
