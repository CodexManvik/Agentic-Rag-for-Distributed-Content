#!/usr/bin/env python3
"""
Quick test script to validate Phase 2 session API endpoints
"""
import requests
import json
import sys
from typing import Dict, Any

BASE_URL = "http://127.0.0.1:8000"

def test_endpoint(method: str, endpoint: str, data: Dict[Any, Any] = None) -> Dict[str, Any]:
    """Test an API endpoint and return response data"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url)
        else:
            return {"error": f"Unsupported method: {method}"}
            
        print(f"✅ {method} {endpoint} - Status: {response.status_code}")
        
        if response.status_code >= 400:
            print(f"❌ Error Response: {response.text}")
            return {"error": response.text, "status_code": response.status_code}
        
        if response.headers.get('content-type', '').startswith('application/json'):
            return response.json()
        else:
            return {"response_text": response.text}
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection Error: Server not running at {BASE_URL}")
        return {"error": "Connection failed"}
    except Exception as e:
        print(f"❌ Request Error: {str(e)}")
        return {"error": str(e)}

def main():
    print("🧪 Testing Phase 2 Session API Endpoints")
    print(f"Server: {BASE_URL}")
    print("=" * 50)
    
    # Test 1: Create a new session
    print("\n1️⃣ Creating a new session...")
    session_data = {
        "user_id": "test-user-123",
        "knowledge_base_id": "test-kb",
        "session_metadata": {"test": "phase2", "name": "Test Session"}
    }
    create_result = test_endpoint("POST", "/sessions", session_data)
    
    if "error" in create_result:
        print("❌ Session creation failed, skipping remaining tests")
        return
    
    session_id = create_result.get("session_id")
    print(f"✅ Session created with ID: {session_id}")
    
    # Test 2: Get session details
    print("\n2️⃣ Retrieving session details...")
    get_result = test_endpoint("GET", f"/sessions/{session_id}")
    if "error" not in get_result:
        print(f"✅ Session name: {get_result.get('name')}")
        print(f"✅ Session user: {get_result.get('user_id')}")
    
    # Test 3: List sessions
    print("\n3️⃣ Listing all sessions...")
    list_result = test_endpoint("GET", "/sessions")
    if "error" not in list_result:
        sessions = list_result.get("sessions", [])
        print(f"✅ Found {len(sessions)} session(s)")
    
    # Test 4: Get session stats
    print("\n4️⃣ Getting session statistics...")
    stats_result = test_endpoint("GET", "/sessions/stats")
    if "error" not in stats_result:
        print(f"✅ Total sessions: {stats_result.get('total_sessions', 0)}")
        print(f"✅ Active sessions: {stats_result.get('active_sessions', 0)}")
    
    # Test 5: Update session
    print("\n5️⃣ Updating session...")
    update_data = {
        "session_metadata": {"test": "phase2", "updated": True, "name": "Updated Test Session"}
    }
    update_result = test_endpoint("PUT", f"/sessions/{session_id}", update_data)
    if "error" not in update_result:
        print(f"✅ Session updated, metadata: {update_result.get('session_metadata', {})}")
    
    # Test 6: Test chat with session
    print("\n6️⃣ Testing chat with session context...")
    chat_url = f"/chat/stream?query=Hello&session_id={session_id}"
    try:
        response = requests.get(f"{BASE_URL}{chat_url}", stream=True)
        print(f"✅ GET {chat_url} - Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Session-aware chat endpoint is accessible")
        else:
            print(f"❌ Chat endpoint error: {response.text}")
    except Exception as e:
        print(f"❌ Chat test error: {str(e)}")
    
    # Test 7: Clean up - delete session
    print("\n7️⃣ Cleaning up - deleting session...")
    delete_result = test_endpoint("DELETE", f"/sessions/{session_id}")
    if "error" not in delete_result:
        print("✅ Session deleted successfully")
    
    print("\n🎉 Phase 2 Session API testing complete!")

if __name__ == "__main__":
    main()