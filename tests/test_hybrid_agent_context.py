#!/usr/bin/env python3
"""
Unit tests for HybridAgentContextManager

Tests dual persistence with Agent Framework threads and custom routing analytics.
"""

import unittest
import asyncio
import sys
import os
import json
import tempfile
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from modules.hybrid_agent_context import (
    HybridAgentContextManager,
    ConversationAnalytics
)


class TestConversationAnalytics(unittest.TestCase):
    """Unit tests for ConversationAnalytics class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analytics = ConversationAnalytics()
    
    def test_initialization(self):
        """Test analytics initialization."""
        self.assertEqual(len(self.analytics.exchanges), 0)
        self.assertEqual(len(self.analytics.source_counts), 0)
        self.assertEqual(len(self.analytics.response_times), 0)
    
    def test_record_exchange(self):
        """Test recording an exchange."""
        self.analytics.record_exchange('local', 0.15)
        self.analytics.record_exchange('cloud', 0.85)
        
        self.assertEqual(len(self.analytics.exchanges), 2)
        self.assertEqual(self.analytics.source_counts['local'], 1)
        self.assertEqual(self.analytics.source_counts['cloud'], 1)
    
    def test_get_distribution(self):
        """Test getting source distribution."""
        self.analytics.record_exchange('local', 0.1)
        self.analytics.record_exchange('local', 0.1)
        self.analytics.record_exchange('cloud', 0.5)
        
        dist = self.analytics.get_distribution()
        self.assertAlmostEqual(dist['local'], 66.67, places=1)
        self.assertAlmostEqual(dist['cloud'], 33.33, places=1)
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        self.analytics.record_exchange('local', 0.1)
        self.analytics.record_exchange('local', 0.2)
        self.analytics.record_exchange('local', 0.15)
        
        metrics = self.analytics.get_performance_metrics()
        self.assertIn('local', metrics)
        self.assertAlmostEqual(metrics['local']['avg'], 0.15, places=2)
        self.assertEqual(metrics['local']['min'], 0.1)
        self.assertEqual(metrics['local']['max'], 0.2)
    
    def test_count_switches(self):
        """Test counting model switches."""
        self.analytics.record_exchange('local', 0.1)
        self.analytics.record_exchange('local', 0.1)
        self.analytics.record_exchange('cloud', 0.5)
        self.analytics.record_exchange('local', 0.1)
        
        switches = self.analytics.count_switches()
        self.assertEqual(switches, 2)
    
    def test_get_summary(self):
        """Test getting complete analytics summary."""
        self.analytics.record_exchange('local', 0.1)
        self.analytics.record_exchange('cloud', 0.5)
        
        summary = self.analytics.get_summary()
        self.assertIn('source_counts', summary)
        self.assertIn('total_exchanges', summary)
        self.assertIn('distribution', summary)
        self.assertIn('performance', summary)
        self.assertIn('switches', summary)
        self.assertEqual(summary['total_exchanges'], 2)


class TestHybridAgentContextManager(unittest.TestCase):
    """Unit tests for HybridAgentContextManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.session_id = "test_hybrid_session"
        self.manager = HybridAgentContextManager(self.session_id)
    
    def tearDown(self):
        """Clean up after tests."""
        self.manager = None
    
    def test_initialization(self):
        """Test HybridAgentContextManager initialization."""
        self.assertEqual(self.manager.session_id, self.session_id)
        self.assertIsNone(self.manager.agent_thread)
        self.assertIsNone(self.manager.agent_instance)
        self.assertEqual(len(self.manager.routing_history), 0)
        self.assertEqual(len(self.manager.messages_openai_format), 0)
    
    def test_initialize_agent_thread(self):
        """Test initializing Agent Framework thread."""
        mock_agent = Mock()
        mock_thread = Mock()
        mock_agent.get_new_thread.return_value = mock_thread
        
        async def test_init():
            thread = await self.manager.initialize_agent_thread(mock_agent)
            self.assertEqual(thread, mock_thread)
            self.assertEqual(self.manager.agent_thread, mock_thread)
            self.assertEqual(self.manager.agent_instance, mock_agent)
            mock_agent.get_new_thread.assert_called_once()
        
        asyncio.run(test_init())
    
    def test_add_exchange_with_agent(self):
        """Test adding exchange using Agent Framework."""
        mock_agent = Mock()
        mock_thread = Mock()
        mock_thread.messages = []
        mock_response = Mock(text="AI response")
        mock_agent.get_new_thread.return_value = mock_thread
        mock_agent.run = AsyncMock(return_value=mock_response)
        
        async def test_exchange():
            response_text, response_time = await self.manager.add_exchange_with_agent(
                agent=mock_agent,
                prompt="Test question",
                source="foundry",
                metadata={"complexity": "high"}
            )
            
            self.assertEqual(response_text, "AI response")
            # Response time should be >= 0 (can be 0 for mock)
            self.assertGreaterEqual(response_time, 0)
            self.assertEqual(len(self.manager.routing_metadata), 1)
            self.assertEqual(self.manager.routing_metadata[0]['source'], 'foundry')
        
        asyncio.run(test_exchange())
    
    def test_add_exchange_with_local(self):
        """Test adding exchange from local model."""
        self.manager.add_exchange_with_local(
            prompt="Hello",
            response="Hi there",
            response_time=0.08,
            metadata={"model": "phi-3.5"}
        )
        
        self.assertEqual(len(self.manager.messages_openai_format), 2)
        self.assertEqual(len(self.manager.routing_metadata), 1)
        self.assertEqual(self.manager.routing_metadata[0]['source'], 'local')
        self.assertEqual(self.manager.messages_openai_format[0]['role'], 'user')
        self.assertEqual(self.manager.messages_openai_format[1]['role'], 'assistant')
    
    def test_add_exchange_generic(self):
        """Test adding generic exchange."""
        self.manager.add_exchange_generic(
            prompt="Test query",
            response="Test response",
            source="apim",
            response_time=0.5,
            metadata={"api": "azure"}
        )
        
        self.assertEqual(len(self.manager.routing_metadata), 1)
        self.assertEqual(self.manager.routing_metadata[0]['source'], 'apim')
    
    def test_get_messages_for_local_model(self):
        """Test getting messages for local model."""
        self.manager.add_exchange_with_local("Q1", "A1", 0.1, {})
        self.manager.add_exchange_with_local("Q2", "A2", 0.1, {})
        
        messages = self.manager.get_messages_for_local_model(max_messages=2)
        self.assertEqual(len(messages), 2)  # Only last 2 messages
    
    def test_get_routing_summary(self):
        """Test getting routing summary."""
        self.manager.add_exchange_with_local("Q1", "A1", 0.1, {})
        self.manager.add_exchange_generic("Q2", "A2", "cloud", 0.5, {})
        
        summary = self.manager.get_routing_summary()
        self.assertIn('total_exchanges', summary)
        self.assertIn('routing_distribution', summary)
        self.assertIn('model_switches', summary)
        self.assertEqual(summary['total_exchanges'], 2)
        self.assertEqual(summary['model_switches'], 1)
    
    def test_persist_to_storage(self):
        """Test persisting conversation to storage."""
        self.manager.add_exchange_with_local("Test", "Response", 0.1, {})
        
        async def test_persist():
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                temp_file = f.name
            
            try:
                success = await self.manager.persist_to_storage(temp_file)
                self.assertTrue(success)
                self.assertTrue(os.path.exists(temp_file))
                
                # Verify content
                with open(temp_file, 'r') as f:
                    data = json.load(f)
                    self.assertIn('session_id', data)
                    self.assertIn('routing_metadata', data)
                    self.assertIn('analytics', data)
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        asyncio.run(test_persist())
    
    def test_resume_from_storage(self):
        """Test resuming conversation from storage."""
        # Create test data
        self.manager.add_exchange_with_local("Q1", "A1", 0.1, {})
        
        async def test_resume():
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                temp_file = f.name
            
            try:
                # Persist first
                await self.manager.persist_to_storage(temp_file)
                
                # Create new manager and resume
                new_manager = HybridAgentContextManager("resumed_session")
                success = await new_manager.resume_from_storage(temp_file)
                
                self.assertTrue(success)
                self.assertEqual(len(new_manager.routing_metadata), 1)
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        asyncio.run(test_resume())


if __name__ == '__main__':
    unittest.main()
