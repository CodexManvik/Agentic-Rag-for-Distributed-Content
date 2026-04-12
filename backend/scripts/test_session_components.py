#!/usr/bin/env python3
"""
Test script for session management components.

Verifies functionality of SessionManager, SessionState, ConversationHistory,
and SessionKVCache components.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add app to Python path
backend_root = Path(__file__).parent.parent
app_root = backend_root / "app"
sys.path.insert(0, str(backend_root))

from app.session import (
    SessionManager, 
    SessionState, 
    ConversationHistory, 
    ConversationMessage, 
    MessageRole, 
    ModelConfig,
    SessionKVCache
)

def test_session_state():
    """Test SessionState functionality."""
    print("Testing SessionState...")
    
    # Create model config
    config = ModelConfig(
        backend="llama.cpp",
        model_name="llama-2-7b",
        temperature=0.7,
        context_length=4096
    )
    
    # Create session state
    session = SessionState(
        session_id="test-session-1",
        user_id="test-user",
        knowledge_base_id="test-kb",
        model_config=config
    )
    
    # Test message addition
    message1 = ConversationMessage(MessageRole.USER, "Hello, how are you?")
    session.add_message(message1)
    
    message2 = ConversationMessage(MessageRole.ASSISTANT, "I'm doing well, thank you!")
    session.add_message(message2)
    
    # Test serialization
    serialized = session.to_dict()
    deserialized = SessionState.from_dict(serialized)
    
    assert deserialized.session_id == session.session_id
    assert len(deserialized.conversation_history) == 2
    assert deserialized.model_config.model_name == "llama-2-7b"
    
    print("SessionState tests passed")


def test_conversation_history():
    """Test ConversationHistory functionality."""
    print("Testing ConversationHistory...")
    
    history = ConversationHistory()
    
    # Add messages
    history.add_message(MessageRole.SYSTEM, "You are a helpful assistant.")
    history.add_message(MessageRole.USER, "What is AI?")
    history.add_message(MessageRole.ASSISTANT, "AI stands for Artificial Intelligence...")
    history.add_message(MessageRole.USER, "Tell me more about machine learning.")
    
    # Test filtering
    user_messages = history.get_messages(role_filter=MessageRole.USER)
    assert len(user_messages) == 2
    
    # Test context window management
    context_messages = history.get_context_messages(max_tokens=100)  # Small budget
    assert len(context_messages) >= 1  # Should at least include system message
    
    # Test truncation
    truncated = history.truncate_to_context_length(context_length=200)
    assert len(truncated) <= 4
    
    # Test serialization
    serialized = history.serialize()
    deserialized = ConversationHistory.deserialize(serialized)
    assert len(deserialized.messages) == len(history.messages)
    
    print("ConversationHistory tests passed")


def test_session_manager():
    """Test SessionManager functionality."""
    print("Testing SessionManager...")
    
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "test_sessions.db")
        manager = SessionManager(db_path=db_path)
        
        try:
            # Create session
            session = manager.create_session(
                user_id="test-user",
                knowledge_base_id="test-kb"
            )
            
            # Retrieve session
            retrieved = manager.get_session(session.session_id)
            assert retrieved is not None
            assert retrieved.session_id == session.session_id
            assert retrieved.user_id == "test-user"
            
            # Update session
            session.add_message(ConversationMessage(MessageRole.USER, "Test message"))
            success = manager.update_session(session)
            assert success
            
            # List sessions
            sessions = manager.list_sessions(user_id="test-user")
            assert len(sessions) == 1
            
            # Delete session
            deleted = manager.delete_session(session.session_id)
            assert deleted
            
            # Verify deletion
            retrieved_after_delete = manager.get_session(session.session_id)
            assert retrieved_after_delete is None
            
        finally:
            # Ensure cleanup
            manager.close()
            
            # On Windows, add a small delay to ensure file handles are released
            import time
            time.sleep(0.1)
    
    print("SessionManager tests passed")


def test_kv_cache():
    """Test SessionKVCache functionality."""
    print("Testing SessionKVCache...")
    
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = SessionKVCache(
            session_id="test-session-kv",
            cache_dir=temp_dir
        )
        
        # Test save state
        test_tokens = [1, 2, 3, 4, 5]
        test_metadata = {"model": "test-model", "temperature": 0.7}
        
        success = cache.save_state(
            context_tokens=test_tokens,
            kv_state={"test": "data"},
            metadata=test_metadata
        )
        assert success
        
        # Test load state
        loaded = cache.load_state()
        assert loaded is not None
        assert loaded["context_tokens"] == test_tokens
        assert loaded["metadata"] == test_metadata
        
        # Test context reuse check
        similar_tokens = [1, 2, 3, 4, 6]  # 80% overlap
        can_reuse = cache.can_reuse_context(similar_tokens, min_overlap=0.8)
        assert can_reuse
        
        different_tokens = [10, 20, 30]  # No overlap
        cannot_reuse = cache.can_reuse_context(different_tokens, min_overlap=0.8)
        assert not cannot_reuse
        
        # Test cache info
        info = cache.get_cache_info()
        assert info["has_cache"]
        assert info["token_count"] == 5
        
        # Test clear
        cleared = cache.clear()
        assert cleared
        
        info_after_clear = cache.get_cache_info()
        assert not info_after_clear["has_cache"]
        assert info_after_clear["token_count"] == 0
    
    print("SessionKVCache tests passed")


def test_integration():
    """Test integration between session components."""
    print("Testing session component integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "integration_sessions.db")
        cache_dir = os.path.join(temp_dir, "kv_cache")
        
        # Initialize components
        manager = SessionManager(db_path=db_path)
        
        # Create session with model config
        config = ModelConfig(
            backend="llama.cpp",
            model_name="test-model",
            context_length=2048
        )
        
        session = manager.create_session(
            user_id="integration-user",
            knowledge_base_id="integration-kb",
            model_config=config
        )
        
        # Add conversation history
        session.add_message(ConversationMessage(MessageRole.SYSTEM, "You are helpful."))
        session.add_message(ConversationMessage(MessageRole.USER, "Hello!"))
        session.add_message(ConversationMessage(MessageRole.ASSISTANT, "Hi there!"))
        
        # Update session in manager
        manager.update_session(session)
        
        # Initialize KV cache for session
        kv_cache = SessionKVCache(session.session_id, cache_dir)
        kv_cache.save_state(
            context_tokens=[100, 200, 300],
            metadata={"integration": "test"}
        )
        
        # Retrieve and verify session
        retrieved = manager.get_session(session.session_id)
        assert len(retrieved.conversation_history) == 3
        assert retrieved.model_config.context_length == 2048
        
        # Verify KV cache works with session
        cache_info = kv_cache.get_cache_info()
        assert cache_info["session_id"] == session.session_id
        assert cache_info["has_cache"]
        
        print("Integration tests passed")


def main():
    """Run all session component tests."""
    print("Starting Session Management Tests...\n")
    
    try:
        test_session_state()
        print()
        
        test_conversation_history()
        print()
        
        test_session_manager()
        print()
        
        test_kv_cache()
        print()
        
        test_integration()
        print()
        
        print("=" * 50)
        print("All session management tests passed!")
        print("=" * 50)
        
        return 0
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())