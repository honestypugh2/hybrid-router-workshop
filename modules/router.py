"""
Intelligent Query Routing Module

This module implements the core routing logic for the hybrid LLM system,
determining whether queries should be processed locally or in the cloud.
"""

import re
import time
from typing import Tuple, Dict, Any
from enum import Enum
from dataclasses import dataclass


class ModelTarget(Enum):
    """Available model targets for routing"""
    LOCAL = "local"
    CLOUD = "cloud"


@dataclass
class QueryAnalysis:
    """Results of query complexity analysis"""
    word_count: int
    complexity_score: float
    has_complex_keywords: bool
    requires_reasoning: bool
    estimated_tokens: int
    analysis_time: float


class HybridRouter:
    """
    Intelligent router that decides between local and cloud models
    based on query characteristics and complexity analysis.
    """
    
    def __init__(self, complexity_threshold: float = 0.5):
        """
        Initialize the hybrid router.
        
        Args:
            complexity_threshold: Threshold above which queries route to cloud
        """
        self.complexity_threshold = complexity_threshold
        self.routing_stats = {
            'total_queries': 0,
            'local_routes': 0,
            'cloud_routes': 0,
            'analysis_time_total': 0.0
        }
        
        # Keywords that typically indicate complex queries requiring cloud processing
        self.complex_keywords = {
            'analyze', 'analysis', 'explain', 'summarize', 'summary',
            'compare', 'comparison', 'evaluate', 'assessment', 'review',
            'strategy', 'plan', 'planning', 'recommend', 'recommendation',
            'detailed', 'comprehensive', 'thorough', 'in-depth',
            'research', 'investigate', 'examine', 'study',
            'create', 'generate', 'write', 'compose', 'draft',
            'code', 'program', 'algorithm', 'implement',
            'translate', 'conversion', 'transform'
        }
        
        # Keywords that typically indicate simple queries suitable for local processing
        self.simple_keywords = {
            'hello', 'hi', 'hey', 'thanks', 'thank you',
            'what', 'when', 'where', 'who', 'how',
            'is', 'are', 'can', 'will', 'do', 'does',
            'define', 'definition', 'meaning', 'means'
        }

    def analyze_query_characteristics(self, query: str) -> QueryAnalysis:
        """
        Analyze query characteristics to determine complexity.
        
        Args:
            query: User query string
            
        Returns:
            QueryAnalysis object with detailed analysis results
        """
        start_time = time.time()
        
        # Basic metrics
        word_count = len(query.split())
        query_lower = query.lower()
        
        # Check for complex keywords
        has_complex_keywords = any(keyword in query_lower for keyword in self.complex_keywords)
        
        # Check for reasoning indicators
        reasoning_patterns = [
            r'\bwhy\b.*\?',  # Why questions
            r'\bhow.*works?\b',  # How it works questions
            r'\bcompare\b.*\band\b',  # Comparison requests
            r'\bpros\s+and\s+cons\b',  # Pros and cons
            r'\bdifference\s+between\b',  # Difference questions
            r'\bstep.*by.*step\b',  # Step-by-step requests
        ]
        requires_reasoning = any(re.search(pattern, query_lower) for pattern in reasoning_patterns)
        
        # Estimate token count (rough approximation: 1 token ≈ 0.75 words)
        estimated_tokens = int(word_count * 1.33)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(
            word_count, has_complex_keywords, requires_reasoning, query_lower
        )
        
        analysis_time = time.time() - start_time
        
        return QueryAnalysis(
            word_count=word_count,
            complexity_score=complexity_score,
            has_complex_keywords=has_complex_keywords,
            requires_reasoning=requires_reasoning,
            estimated_tokens=estimated_tokens,
            analysis_time=analysis_time
        )

    def _calculate_complexity_score(self, word_count: int, has_complex_keywords: bool,
                                  requires_reasoning: bool, query_lower: str) -> float:
        """Calculate overall complexity score for the query."""
        score = 0.0
        
        # Length-based scoring
        if word_count > 20:
            score += 0.4
        elif word_count > 10:
            score += 0.2
        elif word_count <= 3:
            score -= 0.2
        
        # Keyword-based scoring
        if has_complex_keywords:
            score += 0.3
        
        # Check for simple greeting patterns
        simple_greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon']
        if any(greeting in query_lower for greeting in simple_greetings) and word_count <= 5:
            score -= 0.4
        
        # Reasoning requirement scoring
        if requires_reasoning:
            score += 0.3
        
        # Simple factual questions (often start with what/when/where)
        factual_patterns = [r'^\s*what\s+is\b', r'^\s*when\s+is\b', r'^\s*where\s+is\b']
        if any(re.search(pattern, query_lower) for pattern in factual_patterns) and word_count <= 8:
            score -= 0.1
        
        # Mathematical expressions or calculations
        math_patterns = [r'\d+\s*[\+\-\*\/]\s*\d+', r'\bcalculate\b', r'\bcompute\b']
        if any(re.search(pattern, query_lower) for pattern in math_patterns):
            if word_count <= 10:
                score -= 0.1  # Simple math can stay local
            else:
                score += 0.2  # Complex math analysis goes to cloud
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))

    def route_query(self, query: str, analysis: QueryAnalysis = None) -> Tuple[ModelTarget, str]:
        """
        Route a query to the appropriate model based on complexity analysis.
        
        Args:
            query: User query string
            analysis: Optional pre-computed analysis (will compute if not provided)
            
        Returns:
            Tuple of (target_model, routing_reason)
        """
        if analysis is None:
            analysis = self.analyze_query_characteristics(query)
        
        # Update statistics
        self.routing_stats['total_queries'] += 1
        self.routing_stats['analysis_time_total'] += analysis.analysis_time
        
        # Routing decision logic
        if analysis.complexity_score >= self.complexity_threshold:
            target = ModelTarget.CLOUD
            self.routing_stats['cloud_routes'] += 1
            
            # Determine specific reason for cloud routing
            reasons = []
            if analysis.has_complex_keywords:
                reasons.append("complex keywords detected")
            if analysis.requires_reasoning:
                reasons.append("reasoning required")
            if analysis.word_count > 15:
                reasons.append("lengthy query")
            if analysis.complexity_score > 0.7:
                reasons.append("high complexity score")
            
            reason = f"Cloud routing: {', '.join(reasons) if reasons else 'above complexity threshold'}"
            
        else:
            target = ModelTarget.LOCAL
            self.routing_stats['local_routes'] += 1
            
            # Determine specific reason for local routing
            if analysis.word_count <= 5 and any(word in query.lower() for word in self.simple_keywords):
                reason = "Local routing: simple greeting/question"
            elif analysis.complexity_score < 0.2:
                reason = "Local routing: very low complexity"
            elif not analysis.has_complex_keywords and not analysis.requires_reasoning:
                reason = "Local routing: straightforward query"
            else:
                reason = "Local routing: below complexity threshold"
        
        return target, reason

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get detailed routing statistics."""
        total = self.routing_stats['total_queries']
        if total == 0:
            return self.routing_stats.copy()
        
        stats = self.routing_stats.copy()
        stats.update({
            'local_percentage': (self.routing_stats['local_routes'] / total) * 100,
            'cloud_percentage': (self.routing_stats['cloud_routes'] / total) * 100,
            'average_analysis_time': self.routing_stats['analysis_time_total'] / total
        })
        return stats

    def reset_statistics(self):
        """Reset all routing statistics."""
        self.routing_stats = {
            'total_queries': 0,
            'local_routes': 0,
            'cloud_routes': 0,
            'analysis_time_total': 0.0
        }

    def adjust_complexity_threshold(self, new_threshold: float):
        """
        Adjust the complexity threshold for routing decisions.
        
        Args:
            new_threshold: New threshold value (0.0 to 1.0)
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError("Complexity threshold must be between 0.0 and 1.0")
        
        self.complexity_threshold = new_threshold


# Convenience functions for backward compatibility
def analyze_query_characteristics(query: str) -> QueryAnalysis:
    """Standalone function for query analysis."""
    router = HybridRouter()
    return router.analyze_query_characteristics(query)


def route_query(query: str, analysis: QueryAnalysis = None) -> Tuple[str, str]:
    """Standalone function for query routing (returns string target for compatibility)."""
    router = HybridRouter()
    target, reason = router.route_query(query, analysis)
    return target.value, reason


# Example usage and testing
if __name__ == "__main__":
    # Create router instance
    router = HybridRouter()
    
    # Test queries
    test_queries = [
        "Hello there!",
        "What's 5 + 3?",
        "Can you analyze the quarterly sales performance and provide recommendations?",
        "Thanks for your help",
        "Write a comprehensive business plan for a tech startup",
        "What does AI stand for?",
        "Compare the advantages and disadvantages of cloud vs on-premise deployment"
    ]
    
    print("🧠 Hybrid Router Test Results")
    print("=" * 50)
    
    for query in test_queries:
        analysis = router.analyze_query_characteristics(query)
        target, reason = router.route_query(query, analysis)
        
        print(f"\n📝 Query: {query}")
        print(f"🎯 Target: {target.value.upper()}")
        print(f"📊 Complexity: {analysis.complexity_score:.3f}")
        print(f"💭 Reason: {reason}")
    
    # Show statistics
    print(f"\n📈 Routing Statistics:")
    stats = router.get_routing_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")