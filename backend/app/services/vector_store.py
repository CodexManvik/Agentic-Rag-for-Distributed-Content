from collections.abc import Iterable
from math import isfinite
import re

import chromadb
from rank_bm25 import BM25Okapi

from app.config import settings
from app.graph.state import RetrievedChunk, RetrievalQuality
from app.services.llm import get_chroma_embedding_function


_client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
_embedding_function = get_chroma_embedding_function()
_collection = _client.get_or_create_collection(
    name=settings.chroma_collection_name,
    embedding_function=_embedding_function,
)


def get_collection():
    return _collection


def reset_collection() -> None:
    global _collection
    try:
        _client.delete_collection(name=settings.chroma_collection_name)
    except Exception:
        pass
    _collection = _client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=_embedding_function,
    )


def _normalize_distance(distance: float) -> float:
    value = 1.0 / (1.0 + max(distance, 0.0))
    if not isfinite(value):
        return 0.0
    return value


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bm25_candidates(query: str, k: int) -> dict[str, float]:
    payload = _collection.get(include=["documents", "metadatas"])
    ids = payload.get("ids", [])
    docs = payload.get("documents", [])

    if not ids or not docs:
        return {}

    corpus_tokens = [_tokenize(doc or "") for doc in docs]
    bm25 = BM25Okapi(corpus_tokens)
    scores = bm25.get_scores(_tokenize(query))
    if len(scores) == 0:
        return {}

    max_score = max(scores)
    denom = max_score if max_score > 0 else 1.0
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return {str(ids[i]): float(scores[i] / denom) for i in ranked_indices}


def query_chunks(queries: Iterable[str], top_k: int | None = None) -> list[RetrievedChunk]:
    final_k = top_k or settings.retrieval_top_k
    per_query_k = settings.retrieval_per_query_k
    bm25_k = settings.retrieval_bm25_k
    merged: dict[str, RetrievedChunk] = {}

    for query in queries:
        vec_result = _collection.query(query_texts=[query], n_results=per_query_k)
        vec_ids = vec_result.get("ids", [[]])[0]
        vec_docs = vec_result.get("documents", [[]])[0]
        vec_metas = vec_result.get("metadatas", [[]])[0]
        vec_distances = vec_result.get("distances", [[]])[0]
        bm25_scores = _bm25_candidates(query, bm25_k)

        for idx, doc_id in enumerate(vec_ids):
            metadata = vec_metas[idx] if idx < len(vec_metas) and vec_metas[idx] else {}
            content = vec_docs[idx] if idx < len(vec_docs) else ""
            distance = float(vec_distances[idx]) if idx < len(vec_distances) else 1.0
            vector_score = _normalize_distance(distance)
            keyword_score = bm25_scores.get(str(doc_id), 0.0)
            score = 0.7 * vector_score + 0.3 * keyword_score

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


def assess_retrieval_adequacy(chunks: list[RetrievedChunk]) -> RetrievalQuality:
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

    adequate = (
        max_score >= settings.retrieval_min_score
        and chunk_count >= settings.retrieval_min_chunks
        and source_diversity >= settings.retrieval_min_source_diversity
    )
    reason = "Adequate evidence" if adequate else "Evidence quality below threshold"
    return {
        "max_score": max_score,
        "avg_score": avg_score,
        "source_diversity": source_diversity,
        "chunk_count": chunk_count,
        "adequate": adequate,
        "reason": reason,
    }
