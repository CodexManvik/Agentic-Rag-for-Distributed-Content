from collections.abc import Iterable
from dataclasses import dataclass
import logging
from math import isfinite
import os
import re
from threading import Lock
from typing import Any

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "")

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.config import settings
from app.graph.state import RetrievedChunk, RetrievalQuality
from app.services.llm import get_shared_chroma_embedding_function


@dataclass
class _BM25Cache:
    ids: list[str]
    docs: list[str]
    bm25: BM25Okapi


def _create_client() -> chromadb.PersistentClient:
    try:
        return chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    except TypeError:
        # Compatibility fallback for older/newer Chroma constructor signatures.
        return chromadb.PersistentClient(path=settings.chroma_persist_directory)


_client = _create_client()
_collection: Any | None = None
_collection_lock = Lock()
_bm25_cache: _BM25Cache | None = None
_bm25_lock = Lock()


def _ensure_collection():
    global _collection
    if _collection is not None:
        return _collection
    with _collection_lock:
        if _collection is None:
            _collection = _client.get_or_create_collection(
                name=settings.chroma_collection_name,
                embedding_function=get_shared_chroma_embedding_function(),
            )
    return _collection


def get_collection():
    return _ensure_collection()


def reset_collection() -> None:
    global _collection
    try:
        _client.delete_collection(name=settings.chroma_collection_name)
    except Exception:
        pass
    _collection = _client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=get_shared_chroma_embedding_function(),
    )
    refresh_bm25_cache()


def refresh_bm25_cache() -> None:
    global _bm25_cache
    if not settings.bm25_cache_enabled or not settings.hybrid_retrieval_enabled:
        _bm25_cache = None
        return

    collection = _ensure_collection()
    payload = collection.get(include=["documents"])
    ids_raw = payload.get("ids", [])
    docs_raw = payload.get("documents", [])
    ids = [str(i) for i in ids_raw]
    docs = [str(d or "") for d in docs_raw]
    if not ids or not docs:
        _bm25_cache = None
        return

    corpus_tokens = [_tokenize(doc) for doc in docs]
    with _bm25_lock:
        _bm25_cache = _BM25Cache(ids=ids, docs=docs, bm25=BM25Okapi(corpus_tokens))


def _normalize_distance(distance: float) -> float:
    value = 1.0 / (1.0 + max(distance, 0.0))
    if not isfinite(value):
        return 0.0
    return value


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "what", "when", "where", "who", "why", "with",
}


def _query_terms(query: str) -> set[str]:
    return {t for t in _tokenize(query) if len(t) > 2 and t not in _STOPWORDS}


def _query_entities(query: str) -> set[str]:
    entities = set(re.findall(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", query))
    acronyms = set(re.findall(r"\b[A-Z]{2,}\b", query))
    return {e.lower() for e in (entities | acronyms)}


def _is_hard_query(query: str, sub_queries: list[str] | None = None) -> bool:
    q = query.lower()
    if any(k in q for k in ["compare", "difference", "versus", "tradeoff", "multi-hop", "cross-source"]):
        return True
    return bool(sub_queries and len(sub_queries) >= 3)


def _bm25_candidates(query: str, k: int) -> dict[str, float]:
    if not settings.hybrid_retrieval_enabled:
        return {}

    cache = _bm25_cache
    if cache is None:
        refresh_bm25_cache()
        cache = _bm25_cache
    if cache is None:
        return {}

    scores = cache.bm25.get_scores(_tokenize(query))
    if len(scores) == 0:
        return {}

    max_score = max(scores)
    denom = max_score if max_score > 0 else 1.0
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return {cache.ids[i]: float(scores[i] / denom) for i in ranked_indices}


def _vector_candidates(query: str, k: int) -> list[dict[str, Any]]:
    collection = _ensure_collection()
    vec_result = collection.query(query_texts=[query], n_results=k)
    vec_ids = vec_result.get("ids", [[]])[0]
    vec_docs = vec_result.get("documents", [[]])[0]
    vec_metas = vec_result.get("metadatas", [[]])[0]
    vec_distances = vec_result.get("distances", [[]])[0]

    rows: list[dict[str, Any]] = []
    for idx, doc_id in enumerate(vec_ids):
        rows.append(
            {
                "doc_id": str(doc_id),
                "content": vec_docs[idx] if idx < len(vec_docs) else "",
                "metadata": vec_metas[idx] if idx < len(vec_metas) and vec_metas[idx] else {},
                "distance": float(vec_distances[idx]) if idx < len(vec_distances) else 1.0,
            }
        )
    return rows


def query_chunks(queries: Iterable[str], top_k: int | None = None) -> list[RetrievedChunk]:
    final_k = top_k or settings.effective_retrieval_top_k
    per_query_k = settings.effective_retrieval_per_query_k
    bm25_k = settings.retrieval_bm25_k
    merged: dict[str, RetrievedChunk] = {}
    vector_weight = max(0.0, settings.vector_weight)
    bm25_weight = max(0.0, settings.bm25_weight if settings.hybrid_retrieval_enabled else 0.0)
    denom = vector_weight + bm25_weight
    if denom <= 0:
        vector_weight = 1.0
        bm25_weight = 0.0
        denom = 1.0
    vector_weight /= denom
    bm25_weight /= denom

    for query in queries:
        vector_rows = _vector_candidates(query, per_query_k)
        bm25_scores = _bm25_candidates(query, bm25_k)

        for row in vector_rows:
            doc_id = row["doc_id"]
            metadata = row["metadata"]
            content = row["content"]
            distance = row["distance"]
            vector_score = _normalize_distance(distance)
            keyword_score = bm25_scores.get(str(doc_id), 0.0)
            score = vector_weight * vector_score + bm25_weight * keyword_score

            if doc_id not in merged:
                merged[doc_id] = {
                    "chunk_id": str(doc_id),
                    "source": str(metadata.get("source", "unknown")),
                    "content": content,
                    "score": score,
                    "metadata": metadata,
                    "matched_subqueries": [query],
                    "relevance_components": {
                        "vector_score": vector_score,
                        "keyword_score": keyword_score,
                        "combined_score": score,
                    },
                }
            else:
                existing = merged[doc_id]
                existing_matches = existing.get("matched_subqueries", [])
                if query not in existing_matches:
                    existing_matches.append(query)
                    existing["matched_subqueries"] = existing_matches
                if score > existing["score"]:
                    existing["score"] = score
                    existing["relevance_components"] = {
                        "vector_score": vector_score,
                        "keyword_score": keyword_score,
                        "combined_score": score,
                    }

    ranked = sorted(merged.values(), key=lambda c: c["score"], reverse=True)
    return ranked[:final_k]


def assess_retrieval_adequacy(
    chunks: list[RetrievedChunk],
    query: str = "",
    sub_queries: list[str] | None = None,
) -> RetrievalQuality:
    if not chunks:
        return {
            "max_score": 0.0,
            "avg_score": 0.0,
            "source_diversity": 0,
            "chunk_count": 0,
            "adequate": False,
            "reason": "No chunks retrieved",
        }

    scores = [float(c["score"]) for c in chunks]
    source_diversity = len({c["source"] for c in chunks})
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    chunk_count = len(chunks)

    hard_query = _is_hard_query(query, sub_queries)
    min_score = settings.retrieval_min_score + (settings.retrieval_hard_query_min_score_boost if hard_query else 0.0)
    min_diversity = max(
        settings.retrieval_min_source_diversity,
        settings.retrieval_hard_query_min_source_diversity if hard_query else settings.retrieval_min_source_diversity,
    )

    query_terms = _query_terms(query)
    query_entities = _query_entities(query)
    corpus_text = "\n".join(str(c["content"]) for c in chunks)
    corpus_terms = set(_tokenize(corpus_text))

    term_overlap_ratio = (len(query_terms & corpus_terms) / len(query_terms)) if query_terms else 1.0
    entity_overlap_ratio = (len(query_entities & corpus_terms) / len(query_entities)) if query_entities else 1.0

    adequate = (
        max_score >= min_score
        and chunk_count >= settings.retrieval_min_chunks
        and source_diversity >= min_diversity
        and term_overlap_ratio >= settings.retrieval_query_overlap_min
        and entity_overlap_ratio >= settings.retrieval_entity_overlap_min
    )

    if adequate:
        reason = "Adequate evidence"
    elif term_overlap_ratio < settings.retrieval_query_overlap_min or entity_overlap_ratio < settings.retrieval_entity_overlap_min:
        reason = "Weak topical match to query intent"
    else:
        reason = "Evidence quality below threshold"
    return {
        "max_score": max_score,
        "avg_score": avg_score,
        "source_diversity": source_diversity,
        "chunk_count": chunk_count,
        "adequate": adequate,
        "reason": reason,
    }
