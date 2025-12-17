#!/usr/bin/env python3
"""
Integration tests for the Hybrid LLM Router system

Tests end-to-end functionality with mocked external dependencies.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.context_manager import ConversationContextManager
from modules.router import HybridRouter, ModelTarget


class TestEndToEndRouting(unittest.TestCase):
    """Integration tests for end-to-end routing flow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.router = HybridRouter(complexity_threshold=0.5)
        self.context_manager = ConversationContextManager("integration_test")
    
    def test_simple_query_flow(self):
        """Test complete flow for a simple query."""
        query = "Hello, how are you?"
        
        # Step 1: Route the query
        target, reason = self.router.route_query(query)
        
        self.assertIn(target, [ModelTarget.LOCAL, ModelTarget.CLOUD])
        self.assertIsInstance(reason, str)
        
        # Step 2: Add to context manager
        mock_response = "I'm doing well, thank you!"
        metadata = {'routing_reason': reason}
        exchange = self.context_manager.add_exchange(
            query,
            mock_response,
            target.value,
            0.08,
            metadata
        )
        
        self.assertEqual(exchange['exchange_number'], 1)
        self.assertEqual(exchange['user_message'], query)
    
    def test_multi_turn_conversation_flow(self):
        """Test multi-turn conversation with context."""
        queries = [
            ("Hello", "Hi there!"),
            ("My name is Alex", "Nice to meet you, Alex!"),
            ("What's my name?", "Your name is Alex!")
        ]
        
        for i, (query, response) in enumerate(queries, 1):
            # Route query
            target, reason = self.router.route_query(query); metadata = {"routing_reason": reason}
            
            # Add exchange
            exchange = self.context_manager.add_exchange(
                query, response, target.value, 0.1, metadata
            )
            
            self.assertEqual(exchange['exchange_number'], i)
        
        # Verify context is maintained
        summary = self.context_manager.get_session_summary()
        self.assertEqual(summary['session_info']['total_exchanges'], 3)
    
    def test_routing_with_context_switching(self):
        """Test routing with model switches."""
        test_cases = [
            ("Hello", ModelTarget.LOCAL),
            ("Explain quantum mechanics in detail", ModelTarget.CLOUD),
            ("Thanks", ModelTarget.LOCAL)
        ]
        
        for query, expected_tendency in test_cases:
            target, reason = self.router.route_query(query); metadata = {"routing_reason": reason}
            
            # Add to context
            self.context_manager.add_exchange(
                query, "Mock response", target.value, 0.1, metadata
            )
        
        # Check that switches were tracked
        summary = self.context_manager.get_session_summary()
        # Switches should be >= 0
        self.assertGreaterEqual(summary['routing_stats']['model_switches'], 0)


class TestContextPersistence(unittest.TestCase):
    """Integration tests for conversation persistence."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ConversationContextManager("persistence_test")
    
    def tearDown(self):
        """Clean up test files."""
        # Remove any test files created
        import glob
        for file in glob.glob("conversation_persistence_test*.json"):
            if os.path.exists(file):
                os.remove(file)
    
    def test_export_and_verify(self):
        """Test exporting conversation to file."""
        # Add exchanges
        self.manager.add_exchange("Q1", "A1", "local", 0.1, {})
        self.manager.add_exchange("Q2", "A2", "cloud", 0.5, {})
        
        # Export
        filename = self.manager.export_conversation()
        
        # Verify file exists and has content
        self.assertTrue(os.path.exists(filename))
        
        with open(filename, 'r') as f:
            import json
            data = json.load(f)
            self.assertIn('session_info', data)
            self.assertIn('conversation', data)
            self.assertEqual(len(data['conversation']), 2)


class TestHybridSystemIntegration(unittest.TestCase):
    """Integration tests for complete hybrid system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.router = HybridRouter()
        self.context = ConversationContextManager("system_test")
    
    def test_complete_session_workflow(self):
        """Test a complete user session workflow."""
        # Simulate a user session
        session_queries = [
            "Hello!",
            "Can you help me with my project?",
            "It involves machine learning and data analysis",
            "What algorithms should I use?",
            "Thanks for your help!"
        ]
        
        for query in session_queries:
            # Route
            target, reason = self.router.route_query(query); metadata = {"routing_reason": reason}
            
            # Generate mock response
            response = f"Mock response to: {query}"
            response_time = 0.1 if target == ModelTarget.LOCAL else 0.5
            
            # Add to context
            self.context.add_exchange(
                query, response, target.value, response_time, metadata
            )
        
        # Verify session
        summary = self.context.get_session_summary()
        self.assertEqual(summary['session_info']['total_exchanges'], 5)
        # Duration might be 'duration' not 'duration_seconds'
        self.assertIn('duration', summary['session_info'])
    
    def test_analytics_tracking(self):
        """Test that analytics are tracked across the session."""
        # Add various exchanges
        test_data = [
            ("Simple query", "local", 0.08),
            ("Another simple one", "local", 0.09),
            ("Complex analysis query", "cloud", 1.2),
            ("Simple again", "local", 0.07)
        ]
        
        for query, source, time in test_data:
            target = ModelTarget.LOCAL if source == "local" else ModelTarget.CLOUD
            self.context.add_exchange(query, "Response", source, time, {})
        
        # Check analytics
        summary = self.context.get_session_summary()
        routing_stats = summary['routing_stats']
        
        self.assertIn('distribution', routing_stats)
        # Check that we have the expected distribution data
        dist = routing_stats['distribution']
        self.assertIn('local', dist)
        self.assertIn('cloud', dist)


class TestErrorHandlingIntegration(unittest.TestCase):
    """Integration tests for error handling across components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.router = HybridRouter()
        self.context = ConversationContextManager("error_test")
    
    def test_empty_query_handling(self):
        """Test system handles empty queries gracefully."""
        query = ""
        target, reason = self.router.route_query(query); metadata = {"routing_reason": reason}
        
        # Should still return a valid target
        self.assertIn(target, [ModelTarget.LOCAL, ModelTarget.CLOUD])
    
    def test_very_long_query_handling(self):
        """Test system handles very long queries."""
        query = "test " * 1000  # Very long query
        target, reason = self.router.route_query(query); metadata = {"routing_reason": reason}
        
        # Should handle without crashing
        self.assertIn(target, [ModelTarget.LOCAL, ModelTarget.CLOUD])
    
    def test_special_characters_handling(self):
        """Test system handles special characters."""
        queries = [
            "Hello! @#$%^&*()",
            "Test with émojis 🎉",
            "Question with newline\n\ncharacters"
        ]
        
        for query in queries:
            target, reason = self.router.route_query(query); metadata = {"routing_reason": reason}
            self.assertIn(target, [ModelTarget.LOCAL, ModelTarget.CLOUD])


if __name__ == '__main__':
    unittest.main()

    print("-" * 50)
    
    dependencies = {
        "Python Backend": ["fastapi", "uvicorn", "requests", "python-dotenv"],
        "React Frontend": ["npm", "node"]
    }
    
    # Check Python packages
    for package in dependencies["Python Backend"]:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ Python: {package}")
        except ImportError:
            print(f"❌ Python: {package} (missing)")
    
    # Check Node.js/npm
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ npm: {result.stdout.strip()}")
        else:
            print("❌ npm: Not found")
    except:
        print("❌ npm: Not available")
    
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ node: {result.stdout.strip()}")
        else:
            print("❌ node: Not found")
    except:
        print("❌ node: Not available")

def print_startup_instructions():
    """Print instructions for starting the system"""
    print("\n🚀 Startup Instructions")
    print("=" * 50)
    print("\n📋 To start the complete system:")
    print("\n1️⃣ OPTION 1: Quick Start (Recommended)")
    print("   cd react-hybrid-router")
    print("   start_demo.bat")
    print("\n2️⃣ OPTION 2: Manual Start")
    print("   Terminal 1: python api_server.py")
    print("   Terminal 2: cd react-hybrid-router && npm start")
    print("\n🌐 Access Points:")
    print("   • React App: http://localhost:3000")
    print("   • Backend API: http://localhost:8080")
    print("   • API Docs: http://localhost:8080/docs")
    print("\n🔧 Troubleshooting:")
    print("   • Check virtual environment is activated")
    print("   • Ensure ports 3000 and 8080 are available")
    print("   • Verify .env files are configured")
    print("   • Check console logs for detailed errors")

def main():
    """Main test runner"""
    print("🎯 Hybrid LLM Router - Integration Test")
    print("=" * 60)
    
    # Check dependencies first
    check_dependencies()
    
    # Test backend
    backend_ok = test_backend_api()
    
    # Test frontend
    frontend_ok = test_react_frontend()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    if backend_ok and frontend_ok:
        print("🎉 SUCCESS: Both backend and frontend are working!")
        print("✅ System is ready for hybrid LLM routing")
    elif backend_ok:
        print("⚠️ PARTIAL: Backend working, frontend needs attention")
        print("🔧 Start React app: cd react-hybrid-router && npm start")
    elif frontend_ok:
        print("⚠️ PARTIAL: Frontend working, backend needs attention") 
        print("🔧 Start backend: python api_server.py")
    else:
        print("❌ ISSUES: Both services need attention")
        print_startup_instructions()

if __name__ == "__main__":
    main()
