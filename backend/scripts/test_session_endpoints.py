#!/usr/bin/env python3
"""
Test script for session management endpoints.
Tests the FastAPI session endpoints through HTTP requests.
"""

import sys
import os
import requests
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

BASE_URL = "http://localhost:8000"
TEST_USER_ID = "test_user"


def print_test_header(test_name: str):
    """Print a formatted test header."""
    print("\n" + "="*50)
    print(f"Testing: {test_name}")
    print("="*50)


def test_create_session():
    """Test creating a new session."""
    print_test_header("Create Session")
    
    data = {
        "user_id": TEST_USER_ID,
        "title": "Test Session",
        "system_prompt": "You are a helpful assistant for testing.",
        "model_config": {
            "model": "llama2",
            "temperature": 0.7,
            "max_tokens": 2048,
            "context_length": 4096
        }
    }
    
    response = requests.post(f"{BASE_URL}/sessions", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        return response.json()["session_id"]
    return None


def test_get_session(session_id: str):
    """Test getting session by ID."""
    print_test_header("Get Session")
    
    response = requests.get(f"{BASE_URL}/sessions/{session_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_update_session(session_id: str):
    """Test updating session."""
    print_test_header("Update Session")
    
    data = {
        "title": "Updated Test Session",
        "metadata": {"updated": True, "test": "value"}
    }
    
    response = requests.put(f"{BASE_URL}/sessions/{session_id}", json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_list_sessions():
    """Test listing sessions."""
    print_test_header("List Sessions")
    
    response = requests.get(f"{BASE_URL}/sessions?user_id={TEST_USER_ID}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_session_stats():
    """Test getting session stats."""
    print_test_header("Session Stats")
    
    response = requests.get(f"{BASE_URL}/sessions/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_chat_with_session(session_id: str):
    """Test chat endpoint with session."""
    print_test_header("Chat with Session")
    
    data = {
        "query": "What is artificial intelligence?",
        "session_id": session_id
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        # Don't print full response as it can be very long
        response_data = response.json()
        print(f"Answer length: {len(response_data.get('answer', ''))}")
        print(f"Citations: {len(response_data.get('citations', []))}")
        print(f"Session ID: {response_data.get('session_id')}")
        print(f"Confidence: {response_data.get('confidence')}")
    else:
        print(f"Error Response: {json.dumps(response.json(), indent=2)}")


def test_delete_session(session_id: str):
    """Test deleting session."""
    print_test_header("Delete Session")
    
    response = requests.delete(f"{BASE_URL}/sessions/{session_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def check_server_health():
    """Check if the server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def main():
    print("Session Management Endpoints Test")
    print(f"Testing against: {BASE_URL}")
    print(f"Current time: {datetime.now()}")
    
    # Check if server is running
    if not check_server_health():
        print("\nERROR: Server is not running or not responding")
        print("Please start the server with: uvicorn app.main:app --reload")
        return 1
    
    print("\nServer is running! Starting tests...\n")
    
    # Run tests in sequence
    session_id = test_create_session()
    
    if session_id:
        test_get_session(session_id)
        test_update_session(session_id)
        test_list_sessions()
        test_session_stats()
        test_chat_with_session(session_id)
        test_delete_session(session_id)
    else:
        print("\nFailed to create session, skipping dependent tests")
        return 1
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)
    return 0


if __name__ == "__main__":
    sys.exit(main())