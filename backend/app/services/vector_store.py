from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
import os
import re
from threading import Lock
from typing import Any

from cachetools import TTLCache, cached
from loguru import logger

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "")

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.config import settings
from app.graph.state import RetrievedChunk, RetrievalQuality
from app.services.llm import get_shared_chroma_embedding_function

# Initialize spaCy for better tokenization and NLP
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except (ImportError, OSError):
    _SPACY_AVAILABLE = False
    logger.warning("spaCy not available. Falling back to regex tokenization. Run: python -m spacy download en_core_web_sm")


@dataclass
class _BM25Cache:
    ids: list[str]
    docs: list[str]
    bm25: BM25Okapi


def _create_client() -> Any:
    persist_dir = settings.resolved_chroma_persist_directory
    try:
        os.makedirs(persist_dir, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to initialize ChromaDB: Cannot create or access directory '{persist_dir}'. "
            f"Ensure the path exists and has proper read/write permissions. "
            f"Original error: {exc}"
        ) from exc
    
    try:
        return chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    except TypeError:
        # Fallback for older chromadb versions that don't support ChromaSettings
        return chromadb.PersistentClient(path=persist_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize ChromaDB client at '{persist_dir}'. "
            f"Verify the directory is accessible and ChromaDB is properly installed. "
            f"Original error: {exc}"
        ) from exc


_client = _create_client()
_collection: Any | None = None
_collection_lock = Lock()
_bm25_cache: _BM25Cache | None = None
_bm25_lock = Lock()


def _ensure_collection() -> Any:
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


def build_bm25_index() -> None:
    refresh_bm25_cache()


def _normalize_distance(distance: float) -> float:
    value = 1.0 / (1.0 + max(distance, 0.0))
    if not isfinite(value):
        return 0.0
    return value


def _tokenize(text: str) -> list[str]:
    """Tokenize text with spaCy for better NLP handling, fallback to regex."""
    if _SPACY_AVAILABLE:
        doc = _nlp(text.lower())
        # Filter out stopwords, punctuation, and short tokens
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and not token.is_punct and len(token.text) > 2
        ]
        return tokens if tokens else re.findall(r"[a-z0-9]+", text.lower())
    else:
        return re.findall(r"[a-z0-9]+", text.lower())


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / u


# Enhanced stopwords from spaCy's default set
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "what", "when", "where", "who", "why", "with",
    "this", "these", "those", "their", "them", "they", "than", "then", "may", "might", "must",
    "should", "should", "would", "could", "can", "will", "has", "have", "had", "do", "does", "did",
}

_ENTITY_IGNORE = {
    "what", "how", "when", "where", "who", "why", "which",
    "compare", "summarize", "describe", "explain", "list", "show", "provide", "give",
    "does", "do", "is", "are", "can", "should", "would",
}

_WORKFLOW_INTENT_TERMS = {
    "workflow", "workflows", "stage", "stages", "pipeline", "agentic", "graph", "langgraph",
    "orchestration", "planner", "retriever", "synthesis", "validator", "this", "project", "system",
    "architecture",
}

_RESEARCH_HINT_TERMS = {
    "paper", "research", "survey", "arxiv", "literature", "citation", "academic",
}


def _query_terms(query: str) -> set[str]:
    return {t for t in _tokenize(query) if len(t) > 2 and t not in _STOPWORDS}


def _query_entities(query: str) -> set[str]:
    entities = set(re.findall(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", query))
    acronyms = set(re.findall(r"\b[A-Z]{2,}\b", query))
    out: set[str] = set()
    for entity in (entities | acronyms):
        lowered = entity.lower()
        if lowered in _STOPWORDS or lowered in _ENTITY_IGNORE:
            continue
        out.add(lowered)
    return out


def _is_hard_query(query: str, sub_queries: list[str] | None = None) -> bool:
    q = query.lower()
    if any(k in q for k in ["compare", "difference", "versus", "tradeoff", "multi-hop", "cross-source"]):
        return True
    return bool(sub_queries and len(sub_queries) >= 3)


def _is_workflow_intent_query(query: str) -> bool:
    terms = _query_terms(query)
    return len(terms & _WORKFLOW_INTENT_TERMS) > 0


def _is_research_query(query: str) -> bool:
    terms = _query_terms(query)
    return len(terms & _RESEARCH_HINT_TERMS) > 0


def _chunk_term_overlap_count(query_terms: set[str], content: str) -> int:
    if not query_terms:
        return 0
    content_terms = set(_tokenize(content))
    return len(query_terms & content_terms)


def _source_boost(query: str, metadata: dict[str, Any], source: str) -> float:
    source_type = str(metadata.get("source_type", "")).lower()
    source_text = " ".join(
        [
            str(source).lower(),
            str(metadata.get("title", "")).lower(),
            str(metadata.get("url", "")).lower(),
            str(metadata.get("path", "")).lower(),
        ]
    )

    boost = 1.0
    workflow_intent = _is_workflow_intent_query(query)
    research_intent = _is_research_query(query)

    if workflow_intent:
        if any(k in source_text for k in ["readme", "architecture", "ideathon", "resource_pack"]):
            boost *= settings.retrieval_workflow_source_boost
        if any(k in source_text for k in ["langgraph", "docs.langchain.com", "python.langchain.com"]):
            boost *= settings.retrieval_langgraph_source_boost
        if source_type == "pdf" and not research_intent:
            boost *= settings.retrieval_pdf_penalty_for_workflow
    return max(0.1, boost)


def _dedupe_similar_chunks(chunks: list[RetrievedChunk], similarity_threshold: float = 0.92) -> list[RetrievedChunk]:
    kept: list[RetrievedChunk] = []
    signatures: list[set[str]] = []
    for chunk in chunks:
        token_set = set(_tokenize(str(chunk.get("content", ""))))
        if not token_set:
            continue
        is_duplicate = False
        for seen in signatures:
            union = len(token_set | seen)
            if union == 0:
                continue
            jaccard = len(token_set & seen) / union
            if jaccard >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(chunk)
            signatures.append(token_set)
    return kept


def _bm25_candidates(query: str, k: int) -> dict[str, float]:
    if not settings.hybrid_retrieval_enabled:
        return {}

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
    query_list = [q for q in queries if str(q).strip()]
    if not query_list:
        return []

    final_k = top_k or settings.effective_retrieval_top_k
    # Pull a larger candidate pool for reranking
    per_query_k = max(settings.effective_retrieval_per_query_k, final_k * 3, 12)
    bm25_k = max(settings.retrieval_bm25_k, final_k * 3, 12)

    merged: dict[str, RetrievedChunk] = {}

    vector_weight = max(0.0, settings.vector_weight)
    bm25_weight = max(0.0, settings.bm25_weight if settings.hybrid_retrieval_enabled else 0.0)

    # New hybrid rerank components
    overlap_weight = float(getattr(settings, "hybrid_weight_content", 0.20))
    meta_weight = float(getattr(settings, "hybrid_weight_meta", 0.10))

    denom = vector_weight + bm25_weight + overlap_weight + meta_weight
    if denom <= 0:
        vector_weight, bm25_weight, overlap_weight, meta_weight, denom = 1.0, 0.0, 0.0, 0.0, 1.0
    vector_weight /= denom
    bm25_weight /= denom
    overlap_weight /= denom
    meta_weight /= denom

    for query in query_list:
        query_terms = _query_terms(query)
        q_token_set = _token_set(query)

        workflow_intent = _is_workflow_intent_query(query)
        min_term_overlap = (
            settings.retrieval_workflow_chunk_min_term_overlap
            if workflow_intent
            else settings.retrieval_chunk_min_term_overlap
        )

        vector_rows = _vector_candidates(query, per_query_k)
        bm25_scores = _bm25_candidates(query, bm25_k)

        # compute maxima for local normalization
        overlap_vals: list[float] = []
        meta_vals: list[float] = []
        for row in vector_rows:
            content = str(row["content"])
            metadata = row["metadata"] if isinstance(row["metadata"], dict) else {}
            source = str(metadata.get("source", "unknown"))
            meta_text = " ".join(
                [
                    source,
                    str(metadata.get("title", "")),
                    str(metadata.get("section", "")),
                    str(metadata.get("url", "")),
                    str(metadata.get("path", "")),
                ]
            )
            overlap_vals.append(_jaccard(q_token_set, _token_set(content)))
            meta_vals.append(_jaccard(q_token_set, _token_set(meta_text)))

        max_overlap = max(overlap_vals) if overlap_vals else 1.0
        max_meta = max(meta_vals) if meta_vals else 1.0
        if max_overlap <= 0:
            max_overlap = 1.0
        if max_meta <= 0:
            max_meta = 1.0

        for i, row in enumerate(vector_rows):
            doc_id = row["doc_id"]
            metadata = row["metadata"] if isinstance(row["metadata"], dict) else {}
            content = str(row["content"])
            distance = float(row["distance"])

            vector_score = _normalize_distance(distance)
            keyword_score = bm25_scores.get(str(doc_id), 0.0)
            overlap_count = _chunk_term_overlap_count(query_terms, content)
            if query_terms and overlap_count < min_term_overlap:
                continue

            content_overlap = overlap_vals[i] / max_overlap
            meta_overlap = meta_vals[i] / max_meta

            source_multiplier = _source_boost(query, metadata, str(metadata.get("source", "unknown")))

            score = (
                vector_weight * vector_score
                + bm25_weight * keyword_score
                + overlap_weight * content_overlap
                + meta_weight * meta_overlap
            ) * source_multiplier

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
                        "content_overlap": content_overlap,
                        "meta_overlap": meta_overlap,
                        "source_boost": source_multiplier,
                        "term_overlap_count": float(overlap_count),
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
                        "content_overlap": content_overlap,
                        "meta_overlap": meta_overlap,
                        "source_boost": source_multiplier,
                        "term_overlap_count": float(overlap_count),
                        "combined_score": score,
                    }

    ranked = sorted(merged.values(), key=lambda c: c["score"], reverse=True)
    deduped = _dedupe_similar_chunks(ranked)
    return deduped[:final_k] if deduped else ranked[:final_k]


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
    # Cap hard-query diversity at 2 — requiring 3 distinct sources in a small corpus causes
    # almost all multi-hop queries to fail adequacy and cascade to false abstains
    min_diversity = max(
        settings.retrieval_min_source_diversity,
        min(2, settings.retrieval_hard_query_min_source_diversity) if hard_query else settings.retrieval_min_source_diversity,
    )

    query_terms = _query_terms(query)
    query_entities = _query_entities(query)
    corpus_text = "\n".join(str(c["content"]) for c in chunks)
    corpus_terms = set(_tokenize(corpus_text))

    term_overlap_ratio = (len(query_terms & corpus_terms) / len(query_terms)) if query_terms else 1.0
    entity_overlap_ratio = (len(query_entities & corpus_terms) / len(query_entities)) if query_entities else 1.0
    top_overlap_counts: list[int] = []
    for chunk in sorted(chunks, key=lambda c: float(c["score"]), reverse=True)[:2]:
        top_overlap_counts.append(_chunk_term_overlap_count(query_terms, str(chunk.get("content", ""))))
    top_relevance_ok = True if not query_terms else any(c >= settings.retrieval_chunk_min_term_overlap for c in top_overlap_counts)

    # Standard adequacy criteria (strict checks)
    strict_adequate = (
        max_score >= min_score
        and chunk_count >= settings.retrieval_min_chunks
        and source_diversity >= min_diversity
        and term_overlap_ratio >= settings.retrieval_query_overlap_min
        and entity_overlap_ratio >= settings.retrieval_entity_overlap_min
        and top_relevance_ok
    )

    # Determine adequacy with explicit branching for different profiles
    adequate: bool
    reason: str
    
    if settings.normalized_runtime_profile == "low_latency":
        # Low-latency mode: use relaxed criteria
        
        # First, check if low_latency_skip_overlap_check allows bypass of strict checks
        if (
            settings.low_latency_skip_overlap_check
            and max_score >= min_score
            and chunk_count >= settings.retrieval_min_chunks
        ):
            adequate = True
            reason = "Adequate evidence (low-latency override)"
        else:
            # Fall back to moderate support criteria
            moderate_support = (
                max_score >= max(0.20, settings.retrieval_min_score * 0.7)
                and chunk_count >= max(2, settings.retrieval_min_chunks)
                and source_diversity >= 1
            )
            if moderate_support:
                adequate = True
                reason = "Adequate evidence (low-latency composite support)"
            else:
                adequate = False
                reason = "Evidence quality below threshold"
    else:
        # Balanced or high-quality mode: use strict criteria
        if strict_adequate:
            adequate = True
            reason = "Adequate evidence"
        elif (
            term_overlap_ratio < settings.retrieval_query_overlap_min
            or entity_overlap_ratio < settings.retrieval_entity_overlap_min
            or not top_relevance_ok
        ):
            adequate = False
            reason = "Weak topical match to query intent"
        else:
            adequate = False
            reason = "Evidence quality below threshold"

    return {
        "max_score": max_score,
        "avg_score": avg_score,
        "source_diversity": source_diversity,
        "chunk_count": chunk_count,
        "adequate": adequate,
        "reason": reason,
    }