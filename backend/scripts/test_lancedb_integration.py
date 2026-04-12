#!/usr/bin/env python3
"""
Test script to verify LanceDB integration is working properly.
"""

import os
import sys

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from loguru import logger
from app.services.retrieval import get_retriever, clear_retrieval_caches
from app.vectorstore.lancedb_store import LanceDBVectorStore


def test_lancedb_connection():
    """Test basic LanceDB connectivity."""
    logger.info("🧪 Testing LanceDB connection...")
    try:
        store = LanceDBVectorStore(db_path="./lancedb_data")
        stats = store.get_stats()
        logger.info(f"✅ LanceDB connection successful: {stats}")
        return True
    except Exception as e:
        logger.error(f"❌ LanceDB connection failed: {e}")
        return False


def test_retrieval_service():
    """Test the retrieval service with LanceDB."""
    logger.info("🧪 Testing retrieval service...")
    try:
        retriever = get_retriever()
        logger.info("✅ Retriever created successfully")
        
        # Test a simple query
        test_query = "What is confluence?"
        results = retriever.retrieve(test_query)
        
        logger.info(f"✅ Query '{test_query}' returned {len(results)} results")
        
        if results:
            first_result = results[0]
            logger.info(f"Sample result: {first_result.text[:100]}...")
        
        return True
    except Exception as e:
        logger.error(f"❌ Retrieval service test failed: {e}")
        return False


def test_knowledge_base_stats():
    """Test knowledge base statistics."""
    logger.info("🧪 Testing knowledge base stats...")
    try:
        store = LanceDBVectorStore(db_path="./lancedb_data")
        
        # Test overall stats
        overall_stats = store.get_stats()
        logger.info(f"Overall stats: {overall_stats}")
        
        # Test hackathon_demo KB stats
        demo_count = store.count_documents("hackathon_demo")
        logger.info(f"Hackathon demo KB has {demo_count} documents")
        
        # List all knowledge bases
        kbs = store.list_knowledge_bases()
        logger.info(f"Available knowledge bases: {kbs}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Knowledge base stats test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("🚀 Starting LanceDB integration tests...")
    
    # Change to backend directory to match the expected working directory
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    
    tests = [
        ("LanceDB Connection", test_lancedb_connection),
        ("Retrieval Service", test_retrieval_service),
        ("Knowledge Base Stats", test_knowledge_base_stats),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info("="*50)
        
        try:
            if test_func():
                passed += 1
                logger.success(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Test Results: {passed} passed, {failed} failed")
    logger.info("="*50)
    
    if failed == 0:
        logger.success("🎉 All tests passed! LanceDB integration is working correctly.")
        return True
    else:
        logger.error(f"⚠️ {failed} test(s) failed. Please check the logs above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)