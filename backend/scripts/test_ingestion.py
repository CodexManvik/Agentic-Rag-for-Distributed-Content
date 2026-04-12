"""
Quick test of LanceDB ingestion with a minimal example.

This script tests the ingestion pipeline with a single web page to verify
that all components are working before running the full resource pack ingestion.
"""

import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from loguru import logger
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from app.vectorstore.lancedb_store import LanceDBVectorStore

# Test configuration
TEST_URL = "https://python.langchain.com/docs/introduction/"
TEST_KB = "test_ingestion"
LANCEDB_PATH = "backend/lancedb_data_test"


def test_basic_ingestion():
    """Test basic ingestion workflow."""
    logger.info("=== Testing LanceDB Ingestion ===")
    
    # Step 1: Initialize embedding model
    logger.info("Loading embedding model...")
    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.success("✓ Embedding model loaded")
    except Exception as exc:
        logger.error(f"✗ Failed to load embedding model: {exc}")
        return False
    
    # Step 2: Initialize vector store
    logger.info("Initializing LanceDB vector store...")
    try:
        vector_store = LanceDBVectorStore(
            db_path=Path(LANCEDB_PATH),
            embedding_function=embedding_function,
            table_name="documents",
        )
        logger.success("✓ Vector store initialized")
    except Exception as exc:
        logger.error(f"✗ Failed to initialize vector store: {exc}")
        return False
    
    # Step 3: Test document addition
    logger.info("Adding test documents...")
    try:
        from langchain_core.documents import Document
        
        # Create simple test documents
        docs = [
            Document(
                page_content="LangChain is a framework for developing applications powered by language models.",
                metadata={"source": "test1", "knowledge_base": TEST_KB},
            ),
            Document(
                page_content="LanceDB is an embedded vector database for AI applications.",
                metadata={"source": "test2", "knowledge_base": TEST_KB},
            ),
        ]
        
        vector_store.add_documents(docs, knowledge_base=TEST_KB)
        logger.success(f"✓ Added {len(docs)} test documents")
    except Exception as exc:
        logger.error(f"✗ Failed to add documents: {exc}")
        return False
    
    # Step 4: Test similarity search
    logger.info("Testing similarity search...")
    try:
        results = vector_store.similarity_search("What is LangChain?", k=2)
        logger.success(f"✓ Search returned {len(results)} results")
        
        if results:
            logger.info(f"First result preview: {results[0]['text'][:100]}...")
    except Exception as exc:
        logger.error(f"✗ Failed to search: {exc}")
        return False
    
    # Step 5: Test knowledge base operations
    logger.info("Testing knowledge base operations...")
    try:
        kbs = vector_store.list_knowledge_bases()
        logger.success(f"✓ Found {len(kbs)} knowledge bases: {kbs}")
        
        if TEST_KB in kbs:
            logger.info(f"Cleaning up test knowledge base: {TEST_KB}")
            vector_store.delete_by_knowledge_base(TEST_KB)
            logger.success("✓ Test KB deleted")
    except Exception as exc:
        logger.error(f"✗ Failed KB operations: {exc}")
        return False
    
    logger.success("=== All tests passed! ===")
    return True


if __name__ == "__main__":
    success = test_basic_ingestion()
    sys.exit(0 if success else 1)
