#!/usr/bin/env python3
"""
Unit tests for ConversationContextManager

Tests conversation history management, context generation,
session tracking, and Agent Framework thread conversion.
"""

import unittest
import asyncio
import sys
import os
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

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

from modules.context_manager import (
    ConversationContextManager,
    MessageRole,
    ModelSource,
    ConversationMessage
)


class TestConversationContextManager(unittest.TestCase):
    """Unit tests for ConversationContextManager class."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.session_id = "test_session_001"
        self.manager = ConversationContextManager(self.session_id)
    
    def tearDown(self):
        """Clean up after each test."""
        self.manager = None
    
    def test_initialization(self):
        """Test ConversationContextManager initialization."""
        self.assertEqual(self.manager.session_id, self.session_id)
        self.assertEqual(len(self.manager.conversation_history), 0)
        self.assertEqual(len(self.manager.chat_history), 0)
        self.assertIsNotNone(self.manager.conversation_stats['start_time'])
    
    def test_add_exchange(self):
        """Test adding a conversation exchange."""
        user_msg = "Hello, how are you?"
        ai_response = "I'm doing well, thank you!"
        source = "local"
        response_time = 0.15
        metadata = {"api_connected": True}
        
        exchange = self.manager.add_exchange(
            user_msg, ai_response, source, response_time, metadata
        )
        
        self.assertEqual(exchange['exchange_number'], 1)
        self.assertEqual(exchange['user_message'], user_msg)
        self.assertEqual(exchange['response'], ai_response)
        self.assertEqual(exchange['source'], source)
        self.assertEqual(exchange['response_time'], response_time)
        self.assertEqual(len(self.manager.chat_history), 1)
        self.assertEqual(len(self.manager.conversation_history), 2)  # user + assistant
    
    def test_multiple_exchanges(self):
        """Test adding multiple exchanges."""
        exchanges = [
            ("Hello", "Hi there!", "local", 0.1),
            ("What's the weather?", "It's sunny.", "cloud", 0.5),
            ("Thanks!", "You're welcome!", "local", 0.08)
        ]
        
        for i, (user, ai, source, time) in enumerate(exchanges, 1):
            exchange = self.manager.add_exchange(user, ai, source, time, {})
            self.assertEqual(exchange['exchange_number'], i)
        
        self.assertEqual(len(self.manager.chat_history), 3)
        self.assertEqual(len(self.manager.conversation_history), 6)  # 3 pairs
    
    def test_model_switches(self):
        """Test tracking model switches."""
        # Add exchanges with different sources
        self.manager.add_exchange("Q1", "A1", "local", 0.1, {})
        self.manager.add_exchange("Q2", "A2", "local", 0.1, {})  # No switch
        self.manager.add_exchange("Q3", "A3", "cloud", 0.5, {})  # Switch!
        self.manager.add_exchange("Q4", "A4", "local", 0.1, {})  # Switch!
        
        summary = self.manager.get_session_summary()
        self.assertEqual(summary['routing_stats']['model_switches'], 2)
    
    def test_get_last_n_messages(self):
        """Test retrieving last N messages."""
        # Add several exchanges
        for i in range(5):
            self.manager.add_exchange(f"Q{i}", f"A{i}", "local", 0.1, {})
        
        # Get last 3 messages
        recent = self.manager.get_last_n_messages(3)
        self.assertLessEqual(len(recent), 3)
        self.assertGreater(len(recent), 0)
    
    def test_get_messages_for_model(self):
        """Test getting messages in OpenAI format for models."""
        self.manager.add_exchange("Hello", "Hi", "local", 0.1, {})
        self.manager.add_exchange("How are you?", "I'm good", "local", 0.1, {})
        
        messages = self.manager.get_messages_for_model('local')
        self.assertGreater(len(messages), 0)
        # Should have user and assistant messages
        roles = [msg.get('role') for msg in messages]
        self.assertIn('user', roles)
        self.assertIn('assistant', roles)
    
    def test_get_conversation_context(self):
        """Test context string generation."""
        self.manager.add_exchange("What is AI?", "AI is artificial intelligence.", "local", 0.1, {})
        
        recent_msgs = self.manager.get_messages_for_model('local')
        context = self.manager.get_conversation_context(recent_msgs)
        
        self.assertIn("What is AI?", context)
        self.assertIn("AI is artificial intelligence", context)
        self.assertIsInstance(context, str)
    
    def test_session_summary(self):
        """Test session summary generation."""
        self.manager.add_exchange("Q1", "A1", "local", 0.1, {})
        self.manager.add_exchange("Q2", "A2", "cloud", 0.5, {})
        
        summary = self.manager.get_session_summary()
        
        self.assertIn('session_info', summary)
        self.assertIn('routing_stats', summary)
        # Check that we have distribution data
        self.assertIn('distribution', summary['routing_stats'])
        self.assertEqual(summary['session_info']['total_exchanges'], 2)
        self.assertEqual(summary['routing_stats']['model_switches'], 1)
    
    def test_export_conversation(self):
        """Test conversation export to JSON."""
        self.manager.add_exchange("Test", "Response", "local", 0.1, {})
        
        filename = self.manager.export_conversation()
        self.assertTrue(os.path.exists(filename))
        
        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)
    
    def test_clear_conversation(self):
        """Test clearing conversation history."""
        self.manager.add_exchange("Q", "A", "local", 0.1, {})
        self.manager.clear_conversation()
        
        self.assertEqual(len(self.manager.conversation_history), 0)
        self.assertEqual(len(self.manager.chat_history), 0)
    
    def test_to_agent_thread_conversion(self):
        """Test conversion to Agent Framework thread."""
        # Add some conversation history
        self.manager.add_exchange("Hello", "Hi there", "local", 0.1, {})
        self.manager.add_exchange("How are you?", "I'm good", "local", 0.1, {})
        
        # Mock agent
        mock_agent = Mock()
        mock_thread = Mock()
        mock_agent.get_new_thread.return_value = mock_thread
        mock_agent.run = AsyncMock(return_value=Mock(text="Response"))
        
        # Test conversion
        async def test_convert():
            thread = await self.manager.to_agent_thread(mock_agent)
            self.assertEqual(thread, mock_thread)
            # Verify agent.run was called for user messages
            self.assertEqual(mock_agent.run.call_count, 2)
        
        asyncio.run(test_convert())


class TestConversationMessage(unittest.TestCase):
    """Unit tests for ConversationMessage dataclass."""
    
    def test_message_creation(self):
        """Test creating a conversation message."""
        msg = ConversationMessage(
            role=MessageRole.USER,
            content="Test message",
            timestamp=datetime.now().isoformat(),
            source=ModelSource.LOCAL,
            response_time=0.1
        )
        
        self.assertEqual(msg.role, MessageRole.USER)
        self.assertEqual(msg.content, "Test message")
        self.assertEqual(msg.source, ModelSource.LOCAL)
    
    def test_to_openai_format(self):
        """Test conversion to OpenAI format."""
        msg = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content="Test response",
            timestamp=datetime.now().isoformat()
        )
        
        openai_msg = msg.to_openai_format()
        self.assertEqual(openai_msg['role'], 'assistant')
        self.assertEqual(openai_msg['content'], 'Test response')


if __name__ == '__main__':
    unittest.main()

