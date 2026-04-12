"""
LanceDB vector store implementation.

Provides vector storage using LanceDB with support for metadata filtering,
hybrid search, and efficient similarity search.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
import uuid
import re

from loguru import logger
from typing import TYPE_CHECKING, Optional, cast, Union

if TYPE_CHECKING:
    from lancedb.table import Table
    from langchain.schema import Document
    from langchain_core.vectorstores import VectorStore
    from langchain.embeddings.base import Embeddings

EmbeddingFunction = Callable[[str], Sequence[float]]
EmbeddingsProvider = Union["Embeddings", EmbeddingFunction]
DocumentLike = Union[dict[str, object], "Document"]
Metadata = dict[str, object]
Vector = Sequence[float]

try:
    import lancedb
    from lancedb.table import Table
    import pyarrow as pa
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False
    logger.warning("lancedb not installed. LanceDB vector store unavailable.")

try:
    from langchain.schema import Document
    from langchain_core.vectorstores import VectorStore
    from langchain.embeddings.base import Embeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class LanceDBVectorStore:
    """
    LanceDB-based vector store for document embeddings.
    
    Provides efficient vector similarity search with metadata filtering
    and multi-knowledge-base support.
    """
    
    def __init__(
        self,
        db_path: Path,
        embedding_function: Optional[EmbeddingsProvider] = None,
        table_name: str = "documents"
    ):
        """
        Initialize LanceDB vector store.
        
        Args:
            db_path: Path to LanceDB database directory
            embedding_function: Function or LangChain embeddings provider to generate embeddings
            table_name: Name of the table to use
        """
        if not LANCEDB_AVAILABLE:
            raise RuntimeError(
                "lancedb is not installed. "
                "Install it with: pip install lancedb"
            )
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.embedding_function = embedding_function
        self.table_name = table_name
        
        # Connect to database
        self.db = lancedb.connect(str(self.db_path))
        self._table: Optional[Table] = None
        
        logger.info(f"Initialized LanceDB at {self.db_path}")

    def _normalize_document(self, document: DocumentLike) -> tuple[str, Metadata]:
        if isinstance(document, dict):
            text = document.get("text", document.get("page_content", ""))
            metadata = document.get("metadata", {})
        else:
            text = getattr(document, "text", None) or getattr(document, "page_content", "")
            metadata = getattr(document, "metadata", {})

        if not isinstance(metadata, dict):
            metadata = dict(metadata)

        if text is None:
            text = ""

        return str(text), cast(Metadata, metadata)

    def _embed_text(self, text: str) -> list[float]:
        if self.embedding_function is None:
            raise ValueError("No embeddings provided and no embedding function set")

        # LlamaIndex embedding adapters
        if hasattr(self.embedding_function, "get_query_embedding"):
            return list(self.embedding_function.get_query_embedding(text))

        if hasattr(self.embedding_function, "get_text_embedding"):
            return list(self.embedding_function.get_text_embedding(text))

        if hasattr(self.embedding_function, "embed_documents"):
            return list(self.embedding_function.embed_documents([text])[0])

        if hasattr(self.embedding_function, "embed_query"):
            return list(self.embedding_function.embed_query(text))

        if callable(self.embedding_function):
            return list(self.embedding_function(text))

        raise TypeError(
            "Unsupported embedding provider. "
            "Provide a callable or an object with embed_documents/embed_query."
        )
    
    def _ensure_table(self) -> Optional[Table]:
        """Ensure table exists and return it."""
        if self._table is None:
            # Check if table exists
            table_names = self.db.table_names()
            
            if self.table_name in table_names:
                self._table = self.db.open_table(self.table_name)
                logger.debug(f"Opened existing table: {self.table_name}")
            else:
                logger.debug(f"Table {self.table_name} does not exist yet")
        
        return self._table

    def _has_searchable_vector_schema(self, table: Table) -> bool:
        """Return True when table has a vector column with vector-search-compatible type."""
        try:
            for field in table.schema:
                if field.name in {"vector", "embedding", "embeddings"}:
                    return pa.types.is_fixed_size_list(field.type)
        except Exception:
            return False
        return False
    
    def add_documents(
        self,
        documents: Sequence[DocumentLike],
        embeddings: Optional[Sequence[Sequence[float]]] = None,
        knowledge_base: str = "default"
    ) -> list[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dicts with 'text' and optional 'metadata'
            embeddings: Pre-computed embeddings (optional, will compute if not provided)
            knowledge_base: Knowledge base identifier for multi-KB support
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Generate IDs
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Prepare data for LanceDB with flattened metadata
        data = []
        
        for i, doc in enumerate(documents):
            text, metadata = self._normalize_document(doc)

            if embeddings and i < len(embeddings):
                vector = list(embeddings[i])
            else:
                vector = self._embed_text(text)

            # Flatten metadata to avoid nested schema issues
            # Extract common fields and store the rest as JSON string
            row = {
                "id": ids[i],
                "text": text,
                "vector": vector,
                "knowledge_base": knowledge_base,
                "source": metadata.get("source", ""),
                "title": metadata.get("title", ""),
                "url": metadata.get("url", ""),
                "file_path": metadata.get("file_path", ""),
                "page": metadata.get("page", 0),
                "chunk_index": metadata.get("chunk_index", 0),
            }

            data.append(row)
        
        # Create or update table
        if self._table is None:
            vector_dim = len(data[0]["vector"]) if data and data[0].get("vector") else 0
            if vector_dim <= 0:
                raise ValueError("Cannot create LanceDB table without non-empty vectors")

            # Define schema to ensure consistency
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), vector_dim)),
                pa.field("knowledge_base", pa.string()),
                pa.field("source", pa.string()),
                pa.field("title", pa.string()),
                pa.field("url", pa.string()),
                pa.field("file_path", pa.string()),
                pa.field("page", pa.int32()),
                pa.field("chunk_index", pa.int32()),
            ])
            self._table = self.db.create_table(self.table_name, data=data, schema=schema)
            logger.info(f"Created table {self.table_name} with {len(data)} documents")
        else:
            self._table.add(data)
            logger.info(f"Added {len(data)} documents to {self.table_name}")
        
        return ids
    
    def similarity_search(
        self,
        query: str,
        query_embedding: Optional[Sequence[float]] = None,
        k: int = 4,
        knowledge_base: Optional[str] = None,
        metadata_filter: Optional[dict[str, object]] = None
    ) -> list[dict[str, object]]:
        """
        Search for similar documents.
        
        Args:
            query: Query text
            query_embedding: Pre-computed query embedding (optional)
            k: Number of results to return
            knowledge_base: Filter by knowledge base (optional)
            metadata_filter: Additional metadata filters (optional)
            
        Returns:
            List of matching documents with scores
        """
        table = self._ensure_table()
        
        if table is None:
            logger.warning("No documents in vector store")
            return []
        
        # Get query embedding
        if query_embedding is None:
            query_embedding = self._embed_text(query)

        # Build filter
        filter_conditions = []
        
        if knowledge_base:
            filter_conditions.append(f"knowledge_base = '{knowledge_base}'")
        
        if metadata_filter:
            for key, value in metadata_filter.items():
                if isinstance(value, str):
                    filter_conditions.append(f"{key} = '{value}'")
                else:
                    filter_conditions.append(f"{key} = {value}")
        
        # Execute search
        try:
            schema_names = []
            try:
                schema_names = [str(name) for name in table.schema.names]
            except Exception:
                schema_names = []

            vector_column = next(
                (name for name in ("vector", "embedding", "embeddings") if name in schema_names),
                None,
            )

            if vector_column:
                search_query = table.search(query_embedding, vector_column_name=vector_column).limit(k)
            else:
                raise ValueError("No vector column in table schema")
            
            if filter_conditions:
                filter_expr = " AND ".join(filter_conditions)
                search_query = search_query.where(filter_expr)
            
            results = search_query.to_list()
            
            # Format results with metadata reconstruction
            formatted_results = []
            for result in results:
                # Reconstruct metadata from flattened fields
                metadata = {
                    "knowledge_base": result.get("knowledge_base", ""),
                    "source": result.get("source", ""),
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "file_path": result.get("file_path", ""),
                    "page": result.get("page", 0),
                    "chunk_index": result.get("chunk_index", 0),
                }
                
                formatted_results.append({
                    "id": result.get("id"),
                    "text": result.get("text"),
                    "metadata": metadata,
                    "score": result.get("_distance", 0.0),  # LanceDB uses _distance
                })
            
            logger.debug(f"Found {len(formatted_results)} results for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")

            # Fallback for legacy tables missing vector columns: lexical retrieval.
            try:
                if hasattr(table, "to_list"):
                    rows = table.to_list()
                else:
                    df = table.to_pandas()
                    rows = df.to_dict(orient="records")
                if filter_conditions:
                    filtered_rows = []
                    for row in rows:
                        keep = True
                        if knowledge_base:
                            keep = str(row.get("knowledge_base", "")) == knowledge_base
                        if keep:
                            filtered_rows.append(row)
                    rows = filtered_rows

                query_terms = set(re.findall(r"\w+", query.lower()))
                ranked: list[tuple[float, dict[str, object]]] = []
                for row in rows:
                    text = str(row.get("text", ""))
                    text_terms = set(re.findall(r"\w+", text.lower()))
                    overlap = len(query_terms.intersection(text_terms))
                    if overlap > 0:
                        ranked.append((float(overlap), row))

                ranked.sort(key=lambda item: item[0], reverse=True)
                lexical_results = []
                for score, row in ranked[:k]:
                    metadata = {
                        "knowledge_base": row.get("knowledge_base", ""),
                        "source": row.get("source", ""),
                        "title": row.get("title", ""),
                        "url": row.get("url", ""),
                        "file_path": row.get("file_path", ""),
                        "page": row.get("page", 0),
                        "chunk_index": row.get("chunk_index", 0),
                    }
                    lexical_results.append(
                        {
                            "id": row.get("id"),
                            "text": row.get("text"),
                            "metadata": metadata,
                            "score": 1.0 / (1.0 + score),
                        }
                    )

                logger.warning(f"Using lexical fallback retrieval with {len(lexical_results)} results")
                return lexical_results
            except Exception as fallback_exc:
                logger.error(f"Fallback search failed: {fallback_exc}")
                return []
    
    def drop_table(self) -> bool:
        """
        Drop the entire table. Use with caution!
        
        Returns:
            True if table was dropped successfully
        """
        try:
            if self.table_name in self.db.table_names():
                self.db.drop_table(self.table_name)
                self._table = None
                logger.info(f"Dropped table {self.table_name}")
                return True
            else:
                logger.info(f"Table {self.table_name} does not exist")
                return True
        except Exception as e:
            logger.error(f"Failed to drop table: {e}")
            return False
    
    def delete_by_knowledge_base(self, knowledge_base: str) -> int:
        """
        Delete all documents from a knowledge base.
        
        Args:
            knowledge_base: Knowledge base identifier
            
        Returns:
            Number of documents deleted
        """
        table = self._ensure_table()
        
        if table is None:
            return 0
        
        try:
            # If legacy schema uses a non-vector-compatible type, rebuild table on reset.
            if not self._has_searchable_vector_schema(table):
                logger.warning("Legacy vector schema detected. Dropping table for clean rebuild.")
                self.drop_table()
                return 0

            # Try to delete by filter (using flattened schema)
            table.delete(f"knowledge_base = '{knowledge_base}'")
            logger.info(f"Deleted documents from knowledge base: {knowledge_base}")
            return 0
            
        except Exception as e:
            # If delete fails due to schema mismatch, drop the entire table
            if "No field named knowledge_base" in str(e):
                logger.warning(f"Schema mismatch detected. Dropping table to recreate with new schema.")
                self.drop_table()
                return 0
            else:
                logger.error(f"Delete failed: {e}")
                return 0
    
    def list_knowledge_bases(self) -> list[str]:
        """
        List all knowledge bases in the store.
        
        Returns:
            List of knowledge base identifiers
        """
        table = self._ensure_table()
        
        if table is None:
            return []
        
        try:
            # Query all unique knowledge bases (using flattened schema)
            results = table.to_pandas()
            
            if results.empty:
                return []
            
            # Extract unique knowledge bases from the knowledge_base column
            if 'knowledge_base' in results.columns:
                kbs = results['knowledge_base'].unique().tolist()
                return sorted([kb for kb in kbs if kb])
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to list knowledge bases: {e}")
            return []
    
    def get_stats(self) -> dict[str, object]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        table = self._ensure_table()
        
        if table is None:
            return {"total_documents": 0, "knowledge_bases": []}
        
        try:
            df = table.to_pandas()
            total_docs = len(df)
            knowledge_bases = self.list_knowledge_bases()
            
            return {
                "total_documents": total_docs,
                "knowledge_bases": knowledge_bases,
                "kb_count": len(knowledge_bases),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_documents": 0, "knowledge_bases": []}
    
    def count_documents(self, knowledge_base: str = None) -> int:
        """
        Count documents in the vector store, optionally filtered by knowledge base.
        
        Args:
            knowledge_base: Optional knowledge base to filter by
            
        Returns:
            Number of documents
        """
        table = self._ensure_table()
        
        if table is None:
            return 0
        
        try:
            df = table.to_pandas()
            if knowledge_base is None:
                return len(df)
            if "knowledge_base" not in df.columns:
                return 0
            return int((df["knowledge_base"] == knowledge_base).sum())
                
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
