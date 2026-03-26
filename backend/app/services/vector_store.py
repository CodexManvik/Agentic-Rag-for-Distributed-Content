from collections.abc import Iterable

import chromadb

from app.config import settings
from app.graph.state import RetrievedChunk


_client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
_collection = _client.get_or_create_collection(name=settings.chroma_collection_name)


def query_chunks(queries: Iterable[str], top_k: int | None = None) -> list[RetrievedChunk]:
    k = top_k or settings.retrieval_top_k
    dedup: dict[str, RetrievedChunk] = {}

    for q in queries:
        result = _collection.query(query_texts=[q], n_results=k)
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        for idx, doc_id in enumerate(ids):
            metadata = metas[idx] if idx < len(metas) and metas[idx] else {}
            content = docs[idx] if idx < len(docs) else ""
            distance = distances[idx] if idx < len(distances) else 1.0
            score = 1.0 - float(distance)

            if doc_id not in dedup:
                dedup[doc_id] = {
                    "chunk_id": doc_id,
                    "source": str(metadata.get("source", "unknown")),
                    "content": content,
                    "score": score,
                    "metadata": metadata,
                }
            elif score > dedup[doc_id]["score"]:
                dedup[doc_id]["score"] = score

    return sorted(dedup.values(), key=lambda c: c["score"], reverse=True)
