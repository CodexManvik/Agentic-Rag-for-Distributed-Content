#!/usr/bin/env python3
"""Test script to check if all imports work without Pydantic errors."""

import sys
import traceback
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(__file__))

try:
    print("Testing imports...")
    
    # Test schemas import first
    print("Importing schemas...")
    from app.api.schemas import CreateSessionRequest, UpdateSessionRequest, SessionStateSchema
    
    # Test main app import
    print("Importing main app...")
    from app.main import app
    
    print("All imports successful! No Pydantic errors detected.")
    
    # Test creating schema instances to verify they work
    print("Testing schema instantiation...")
    
    request = CreateSessionRequest(user_id="test_user")
    print("CreateSessionRequest created successfully")
    
    update_request = UpdateSessionRequest(knowledge_base_id="test_kb")
    print("UpdateSessionRequest created successfully")
    
    print("All tests passed!")

except Exception as e:
    print(f"Import failed: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)