"""LanceDB ingestion service compatible with existing API endpoints."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict

from loguru import logger
from llama_index.core import Document
import fitz  # PyMuPDF
from llama_index.core.node_parser import SentenceSplitter

from app.vectorstore.lancedb_store import LanceDBVectorStore
from app.services.retrieval import clear_retrieval_caches


class IngestionStats(TypedDict):
    """Compatible with existing ingestion stats interface."""
    documents_processed: int
    chunks_added: int
    skipped_duplicates: int
    errors: list[str]


def _create_lancedb_store() -> LanceDBVectorStore:
    """Create LanceDB store instance."""
    return LanceDBVectorStore(db_path="./lancedb_data")


def _process_document_chunks(
    documents: list[Document], 
    store: LanceDBVectorStore, 
    knowledge_base: str = "api_upload"
) -> IngestionStats:
    """Process documents into chunks and add to LanceDB."""
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separator=" "
    )
    
    stats = IngestionStats(
        documents_processed=len(documents),
        chunks_added=0,
        skipped_duplicates=0,
        errors=[]
    )
    
    for doc in documents:
        try:
            # Split document into chunks
            chunks = splitter.split_text(doc.text)
            
            # Convert to LanceDB format
            chunk_docs = []
            for i, chunk_text in enumerate(chunks):
                chunk_doc = {
                    "text": chunk_text,
                    "knowledge_base": knowledge_base,
                    "source": doc.metadata.get("source", "unknown"),
                    "title": doc.metadata.get("title", ""),
                    "url": doc.metadata.get("url", ""),
                    "file_path": doc.metadata.get("file_name", ""),
                    "page": doc.metadata.get("page", 0),
                    "chunk_index": i
                }
                chunk_docs.append(chunk_doc)
            
            # Add to LanceDB
            store.add_documents(chunk_docs)
            stats["chunks_added"] += len(chunk_docs)
            
            logger.debug(f"Added {len(chunk_docs)} chunks from document")
            
        except Exception as e:
            error_msg = f"Error processing document: {str(e)}"
            logger.error(error_msg)
            stats["errors"].append(error_msg)
    
    return stats


def ingest_pdf(file_path: str, knowledge_base: str = "api_upload") -> IngestionStats:
    """
    Ingest a PDF file into LanceDB.
    
    Args:
        file_path: Path to PDF file
        knowledge_base: Knowledge base name for categorization
        
    Returns:
        Ingestion statistics
    """
    logger.info(f"Ingesting PDF: {file_path}")
    
    try:
        store = _create_lancedb_store()
        
        # Use PyMuPDF for better PDF processing
        documents = []
        with fitz.open(file_path) as pdf_doc:
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                text = page.get_text()

                if text.strip():  # Only add pages with content
                    doc = Document(
                        text=text,
                        metadata={
                            "page": page_num + 1,
                            "source": "pdf"
                        }
                    )
                    documents.append(doc)
        
        # Add metadata
        filename = Path(file_path).name
        for doc in documents:
            doc.metadata.update({
                "source": "pdf",
                "file_name": filename,
                "ingested_at": datetime.now(timezone.utc).isoformat()
            })
        
        stats = _process_document_chunks(documents, store, knowledge_base)
        
        # Clear caches after ingestion
        clear_retrieval_caches()
        
        logger.info(f"PDF ingestion complete: {stats['chunks_added']} chunks added")
        return stats
        
    except Exception as e:
        error_msg = f"Failed to ingest PDF {file_path}: {str(e)}"
        logger.error(error_msg)
        return IngestionStats(
            documents_processed=0,
            chunks_added=0,
            skipped_duplicates=0,
            errors=[error_msg]
        )


def ingest_text_file(file_path: str, knowledge_base: str = "api_upload") -> IngestionStats:
    """
    Ingest a text file (.txt, .md) into LanceDB.
    
    Args:
        file_path: Path to text file
        knowledge_base: Knowledge base name for categorization
        
    Returns:
        Ingestion statistics
    """
    logger.info(f"Ingesting text file: {file_path}")
    
    try:
        store = _create_lancedb_store()
        
        # Read text file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        filename = Path(file_path).name
        document = Document(
            text=content,
            metadata={
                "source": "text_file",
                "file_name": filename,
                "ingested_at": datetime.now(timezone.utc).isoformat()
            }
        )
        
        stats = _process_document_chunks([document], store, knowledge_base)
        
        # Clear caches after ingestion
        clear_retrieval_caches()
        
        logger.info(f"Text file ingestion complete: {stats['chunks_added']} chunks added")
        return stats
        
    except Exception as e:
        error_msg = f"Failed to ingest text file {file_path}: {str(e)}"
        logger.error(error_msg)
        return IngestionStats(
            documents_processed=0,
            chunks_added=0,
            skipped_duplicates=0,
            errors=[error_msg]
        )


def reset_knowledge_base(knowledge_base: str) -> bool:
    """
    Reset (delete all documents from) a specific knowledge base.
    
    Args:
        knowledge_base: Knowledge base to reset
        
    Returns:
        True if successful, False otherwise
    """
    try:
        store = _create_lancedb_store()
        store.delete_by_knowledge_base(knowledge_base)
        clear_retrieval_caches()
        logger.info(f"Reset knowledge base: {knowledge_base}")
        return True
    except Exception as e:
        logger.error(f"Failed to reset knowledge base {knowledge_base}: {e}")
        return False


def get_knowledge_base_stats(knowledge_base: Optional[str] = None) -> dict:
    """
    Get statistics for a knowledge base or all knowledge bases.
    
    Args:
        knowledge_base: Optional specific knowledge base
        
    Returns:
        Statistics dictionary
    """
    try:
        store = _create_lancedb_store()
        
        if knowledge_base:
            count = store.count_documents(knowledge_base)
            return {
                "knowledge_base": knowledge_base,
                "document_count": count
            }
        else:
            return store.get_stats()
            
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return {"error": str(e)}
