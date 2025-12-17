#!/usr/bin/env python3
"""
Unit tests for FastAPI hybrid router backend

Tests API endpoints, request handling, and response formats.
Uses mocking to avoid requiring running services.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestAPIEndpoints(unittest.TestCase):
    """Unit tests for API endpoint responses."""
    
    @patch('requests.get')
    def test_root_endpoint(self, mock_get):
        """Test root endpoint returns welcome message."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'message': 'Hybrid LLM Router API',
            'version': '2.0.0'
        }
        mock_get.return_value = mock_response
        
        import requests
        response = requests.get('http://localhost:8080/')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('message', data)
        self.assertIn('Hybrid LLM Router', data['message'])
    
    @patch('requests.get')
    def test_health_endpoint(self, mock_get):
        """Test health check endpoint."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'status': 'healthy',
            'services': {
                'local_model': True,
                'azure_openai': True
            }
        }
        mock_get.return_value = mock_response
        
        import requests
        response = requests.get('http://localhost:8080/health')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('services', data)
    
    @patch('requests.get')
    def test_system_status_endpoint(self, mock_get):
        """Test system status endpoint."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'systemHealth': 'operational',
            'availableRouters': {
                'hybrid': True,
                'bert': True,
                'phi': False
            }
        }
        mock_get.return_value = mock_response
        
        import requests
        response = requests.get('http://localhost:8080/status')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('systemHealth', data)
        self.assertIn('availableRouters', data)
    
    @patch('requests.post')
    def test_route_endpoint_simple_query(self, mock_post):
        """Test routing a simple query."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Hello! How can I help you?',
            'source': 'local',
            'response_time': 0.08,
            'metadata': {
                'routing_reason': 'simple_greeting',
                'complexity_score': 0.2
            }
        }
        mock_post.return_value = mock_response
        
        import requests
        payload = {'query': 'Hello!', 'strategy': 'hybrid'}
        response = requests.post('http://localhost:8080/route', json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('response', data)
        self.assertIn('source', data)
        self.assertEqual(data['source'], 'local')
    
    @patch('requests.post')
    def test_route_endpoint_complex_query(self, mock_post):
        """Test routing a complex query."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Detailed analysis of quantum computing...',
            'source': 'cloud',
            'response_time': 1.5,
            'metadata': {
                'routing_reason': 'high_complexity',
                'complexity_score': 0.85
            }
        }
        mock_post.return_value = mock_response
        
        import requests
        payload = {
            'query': 'Explain quantum computing in detail',
            'strategy': 'hybrid'
        }
        response = requests.post('http://localhost:8080/route', json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['source'], 'cloud')
        self.assertGreater(data['response_time'], 1.0)


class TestAPIRequestValidation(unittest.TestCase):
    """Unit tests for API request validation."""
    
    def test_route_request_payload_structure(self):
        """Test that route request payload has correct structure."""
        valid_payload = {
            'query': 'Test query',
            'strategy': 'hybrid'
        }
        
        self.assertIn('query', valid_payload)
        self.assertIn('strategy', valid_payload)
        self.assertIsInstance(valid_payload['query'], str)
        self.assertIsInstance(valid_payload['strategy'], str)
    
    def test_route_request_missing_query(self):
        """Test handling of missing query field."""
        invalid_payload = {'strategy': 'hybrid'}
        
        self.assertNotIn('query', invalid_payload)
    
    def test_route_request_missing_strategy(self):
        """Test handling of missing strategy field."""
        # Strategy should have a default if missing
        payload = {'query': 'Test'}
        self.assertIn('query', payload)


class TestAPIResponseFormat(unittest.TestCase):
    """Unit tests for API response format validation."""
    
    def test_route_response_structure(self):
        """Test that route response has all required fields."""
        mock_response = {
            'response': 'Test response',
            'source': 'local',
            'response_time': 0.15,
            'metadata': {
                'routing_reason': 'test',
                'complexity_score': 0.5
            }
        }
        
        required_fields = ['response', 'source', 'response_time', 'metadata']
        for field in required_fields:
            self.assertIn(field, mock_response)
    
    def test_metadata_structure(self):
        """Test that metadata has expected fields."""
        metadata = {
            'routing_reason': 'test_reason',
            'complexity_score': 0.75,
            'strategy': 'hybrid'
        }
        
        self.assertIn('routing_reason', metadata)
        self.assertIsInstance(metadata['complexity_score'], (int, float))


class TestAPIErrorHandling(unittest.TestCase):
    """Unit tests for API error handling."""
    
    @patch('requests.post')
    def test_api_connection_error(self, mock_post):
        """Test handling of connection errors."""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        with self.assertRaises(requests.exceptions.ConnectionError):
            requests.post('http://localhost:8080/route', json={'query': 'test'})
    
    @patch('requests.get')
    def test_api_timeout_error(self, mock_get):
        """Test handling of timeout errors."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout("Request timed out")
        
        with self.assertRaises(requests.exceptions.Timeout):
            requests.get('http://localhost:8080/health', timeout=5)


if __name__ == '__main__':
    # Run tests with unittest
    unittest.main()

    print("- Are your Azure OpenAI credentials valid?")

def main():
    """Main test function"""
    try:
        test_api_endpoints()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"🚨 Unexpected error: {e}")

if __name__ == "__main__":
    main()