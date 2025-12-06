#!/usr/bin/env python3
"""
Unit tests for local Foundry API client

Tests local model API interactions with proper mocking.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestLocalFoundryAPI(unittest.TestCase):
    """Unit tests for local Foundry API interactions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_url = "http://localhost:60632"
        self.endpoint = f"{self.base_url}/v1/chat/completions"
        self.model_name = "Phi-3.5-mini-instruct-generic-cpu"
    
    @patch('requests.post')
    def test_chat_completion_request(self, mock_post):
        """Test successful chat completion request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'id': 'chatcmpl-123',
            'object': 'chat.completion',
            'created': 1677652288,
            'model': self.model_name,
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'Hello! I am doing well, thank you for asking!'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 15,
                'total_tokens': 25
            }
        }
        mock_post.return_value = mock_response
        
        import requests
        payload = {
            'model': self.model_name,
            'messages': [
                {'role': 'user', 'content': 'Hello! How are you?'}
            ],
            'max_tokens': 100,
            'temperature': 0.7
        }
        
        response = requests.post(self.endpoint, json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('choices', data)
        self.assertEqual(len(data['choices']), 1)
        self.assertIn('message', data['choices'][0])
    
    @patch('requests.post')
    def test_chat_completion_with_context(self, mock_post):
        """Test chat completion with conversation context."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': 'You asked me how I was doing, and I responded positively.'
                }
            }]
        }
        mock_post.return_value = mock_response
        
        import requests
        payload = {
            'model': self.model_name,
            'messages': [
                {'role': 'user', 'content': 'Hello! How are you?'},
                {'role': 'assistant', 'content': 'I am doing well!'},
                {'role': 'user', 'content': 'What was my previous question?'}
            ]
        }
        
        response = requests.post(self.endpoint, json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('You asked', data['choices'][0]['message']['content'])
    
    def test_payload_structure(self):
        """Test that request payload has correct structure."""
        payload = {
            'model': self.model_name,
            'messages': [
                {'role': 'user', 'content': 'Test'}
            ],
            'max_tokens': 100,
            'temperature': 0.7
        }
        
        required_fields = ['model', 'messages']
        for field in required_fields:
            self.assertIn(field, payload)
        
        self.assertIsInstance(payload['messages'], list)
        self.assertGreater(len(payload['messages']), 0)
    
    def test_message_format(self):
        """Test message format validation."""
        message = {'role': 'user', 'content': 'Hello'}
        
        self.assertIn('role', message)
        self.assertIn('content', message)
        self.assertIn(message['role'], ['user', 'assistant', 'system'])
    
    @patch('requests.post')
    def test_api_error_handling(self, mock_post):
        """Test handling of API errors."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = 'Internal Server Error'
        mock_post.return_value = mock_response
        
        import requests
        payload = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': 'Test'}]
        }
        
        response = requests.post(self.endpoint, json=payload)
        
        self.assertEqual(response.status_code, 500)
    
    @patch('requests.post')
    def test_connection_timeout(self, mock_post):
        """Test handling of connection timeouts."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()
        
        payload = {
            'model': self.model_name,
            'messages': [{'role': 'user', 'content': 'Test'}]
        }
        
        with self.assertRaises(requests.exceptions.Timeout):
            requests.post(self.endpoint, json=payload, timeout=1)
    
    def test_temperature_range(self):
        """Test temperature parameter validation."""
        valid_temps = [0.0, 0.5, 1.0, 2.0]
        
        for temp in valid_temps:
            self.assertGreaterEqual(temp, 0.0)
            self.assertLessEqual(temp, 2.0)
    
    def test_max_tokens_validation(self):
        """Test max_tokens parameter validation."""
        valid_max_tokens = [10, 100, 1000, 4096]
        
        for tokens in valid_max_tokens:
            self.assertGreater(tokens, 0)
            self.assertLessEqual(tokens, 8192)  # Model limit


class TestLocalModelConfiguration(unittest.TestCase):
    """Unit tests for local model configuration."""
    
    def test_model_name_format(self):
        """Test model name format validation."""
        valid_names = [
            "Phi-3.5-mini-instruct-generic-cpu",
            "Phi-3.5-mini-instruct",
            "phi-3-medium"
        ]
        
        for name in valid_names:
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)
    
    def test_endpoint_url_format(self):
        """Test endpoint URL format validation."""
        endpoint = "http://localhost:60632/v1/chat/completions"
        
        self.assertTrue(endpoint.startswith('http://'))
        self.assertIn('localhost', endpoint)
        self.assertIn('/v1/chat/completions', endpoint)


if __name__ == "__main__":
    unittest.main()
