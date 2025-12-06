#!/usr/bin/env python3
"""
Test script to verify telemetry fixes
"""

import sys
import os

# Add modules to path
sys.path.append('.')

def test_telemetry():
    """Test TelemetryCollector functionality"""
    try:
        from modules.telemetry import TelemetryCollector
        print("✅ TelemetryCollector import successful")
        
        telemetry = TelemetryCollector()
        print("✅ TelemetryCollector instantiation successful")
        
        # Test the new method signatures
        session_id = "test_session"
        query_id = "test_query_123"
        
        # Test query received logging
        telemetry.log_query_received(
            session_id=session_id,
            query_id=query_id,
            query="Test query",
            strategy="adaptive"
        )
        print("✅ log_query_received works correctly")
        
        # Test model response logging
        telemetry.log_model_response(
            session_id=session_id,
            query_id=query_id,
            model_endpoint="test_model",
            response="Test response",
            response_time=0.5,
            metadata={"test": True}
        )
        print("✅ log_model_response works correctly")
        
        # Test error logging
        telemetry.log_error(
            session_id=session_id,
            query_id=query_id,
            error_type="TestError",
            error_message="Test error message",
            metadata={"test": True}
        )
        print("✅ log_error works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Telemetry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_azure_manager():
    """Test AzureAIManager initialization"""
    try:
        from modules.azure_ai_manager import AzureAIManager
        print("✅ AzureAIManager import successful")
        
        # This should not throw an error even if Azure resources are missing
        manager = AzureAIManager()
        print("✅ AzureAIManager instantiation completed (graceful degradation)")
        
        return True
        
    except Exception as e:
        print(f"❌ Azure manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Testing telemetry and Azure manager fixes...")
    print()
    
    telemetry_ok = test_telemetry()
    print()
    
    azure_ok = test_azure_manager()
    print()
    
    if telemetry_ok and azure_ok:
        print("✅ All tests passed! The fixes should resolve the React errors.")
    else:
        print("❌ Some tests failed. Check the error messages above.")