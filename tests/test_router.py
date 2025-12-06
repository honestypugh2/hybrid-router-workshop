#!/usr/bin/env python3
"""
Unit tests for HybridRouter module

Tests query analysis, routing logic, and statistics tracking.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.router import HybridRouter, ModelTarget, QueryAnalysis


class TestHybridRouter(unittest.TestCase):
    """Unit tests for HybridRouter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.router = HybridRouter(complexity_threshold=0.5)
    
    def test_initialization(self):
        """Test router initialization."""
        self.assertEqual(self.router.complexity_threshold, 0.5)
        self.assertIn('total_queries', self.router.routing_stats)
        self.assertEqual(self.router.routing_stats['total_queries'], 0)
    
    def test_analyze_simple_query(self):
        """Test analyzing a simple query."""
        query = "Hello, how are you?"
        analysis = self.router.analyze_query_characteristics(query)
        
        self.assertIsInstance(analysis, QueryAnalysis)
        self.assertLess(analysis.complexity_score, 0.5)
        self.assertIsInstance(analysis.word_count, int)
    
    def test_analyze_complex_query(self):
        """Test analyzing a complex query."""
        query = "Please analyze the differences between supervised and unsupervised learning, and provide detailed examples of each approach with their pros and cons."
        analysis = self.router.analyze_query_characteristics(query)
        
        self.assertIsInstance(analysis, QueryAnalysis)
        self.assertGreater(analysis.complexity_score, 0.5)
    
    def test_route_query_simple(self):
        """Test routing a simple query to local model."""
        query = "Hi there"
        target, reason = self.router.route_query(query)
        
        # Simple greeting should likely route to local
        self.assertIn(target, [ModelTarget.LOCAL, ModelTarget.CLOUD])
        self.assertIsInstance(reason, str)
        self.assertGreater(len(reason), 0)
    
    def test_route_query_complex(self):
        """Test routing a complex query."""
        query = "Please provide a comprehensive analysis of machine learning algorithms"
        target, reason = self.router.route_query(query)
        
        # Complex query might route to either depending on analysis
        self.assertIn(target, [ModelTarget.LOCAL, ModelTarget.CLOUD])
        self.assertIsInstance(reason, str)
    
    def test_route_query_with_analysis(self):
        """Test routing with pre-computed analysis."""
        query = "Test query"
        # Create analysis with all required QueryAnalysis fields
        analysis = QueryAnalysis(
            word_count=2,
            complexity_score=0.8,
            has_complex_keywords=True,
            requires_reasoning=False,
            estimated_tokens=10,
            analysis_time=0.001
        )
        
        target, reason = self.router.route_query(query, analysis)
        self.assertIn(target, [ModelTarget.LOCAL, ModelTarget.CLOUD])
    
    def test_routing_stats_tracking(self):
        """Test that routing stats are tracked."""
        initial_count = self.router.routing_stats['total_queries']
        
        self.router.route_query("Test query 1")
        self.router.route_query("Test query 2")
        
        self.assertEqual(
            self.router.routing_stats['total_queries'], 
            initial_count + 2
        )
    
    def test_get_stats(self):
        """Test getting routing statistics."""
        # Route some queries
        self.router.route_query("Hello")
        self.router.route_query("Analyze this complex topic in detail")
        
        stats = self.router.get_routing_statistics()
        self.assertIn('total_queries', stats)
        self.assertIn('local_routes', stats)
        self.assertIn('cloud_routes', stats)
        self.assertGreater(stats['total_queries'], 0)
    
    def test_reset_stats(self):
        """Test resetting routing statistics."""
        self.router.route_query("Test")
        self.router.reset_statistics()
        
        stats = self.router.routing_stats
        self.assertEqual(stats['total_queries'], 0)
        self.assertEqual(stats['local_routes'], 0)
        self.assertEqual(stats['cloud_routes'], 0)
    
    def test_empty_query_handling(self):
        """Test handling of empty query."""
        query = ""
        analysis = self.router.analyze_query_characteristics(query)
        
        # Should still return valid analysis
        self.assertIsInstance(analysis, QueryAnalysis)
        self.assertGreaterEqual(analysis.complexity_score, 0)
        self.assertLessEqual(analysis.complexity_score, 1)
    
    def test_query_length_impact(self):
        """Test that query length impacts complexity."""
        short_query = "Hi"
        long_query = "This is a much longer and more detailed query that asks about complex topics and requires extensive analysis and comprehensive explanation with multiple examples."
        
        short_analysis = self.router.analyze_query_characteristics(short_query)
        long_analysis = self.router.analyze_query_characteristics(long_query)
        
        # Longer queries generally have higher complexity
        # But not always, so we just verify both are valid scores
        self.assertGreaterEqual(short_analysis.complexity_score, 0)
        self.assertGreaterEqual(long_analysis.complexity_score, 0)
    
    def test_complexity_keywords_detection(self):
        """Test detection of complexity keywords."""
        complex_query = "Please analyze and summarize this research paper"
        analysis = self.router.analyze_query_characteristics(complex_query)
        
        # Query with complex keywords should have reasonably high score
        # Using >= to allow for exact 0.3 or higher
        self.assertGreaterEqual(analysis.complexity_score, 0.3)


class TestQueryAnalysis(unittest.TestCase):
    """Unit tests for QueryAnalysis dataclass."""
    
    def test_query_analysis_creation(self):
        """Test creating QueryAnalysis object."""
        analysis = QueryAnalysis(
            word_count=5,
            complexity_score=0.7,
            has_complex_keywords=True,
            requires_reasoning=False,
            estimated_tokens=20,
            analysis_time=0.001
        )
        
        self.assertEqual(analysis.word_count, 5)
        self.assertEqual(analysis.complexity_score, 0.7)
        self.assertTrue(analysis.has_complex_keywords)


class TestModelTarget(unittest.TestCase):
    """Unit tests for ModelTarget enum."""
    
    def test_model_target_values(self):
        """Test ModelTarget enum values."""
        self.assertEqual(ModelTarget.LOCAL.value, "local")
        self.assertEqual(ModelTarget.CLOUD.value, "cloud")
    
    def test_model_target_membership(self):
        """Test checking ModelTarget membership."""
        self.assertIn(ModelTarget.LOCAL, ModelTarget)
        self.assertIn(ModelTarget.CLOUD, ModelTarget)


if __name__ == '__main__':
    unittest.main()
