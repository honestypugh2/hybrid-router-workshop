"""
Phi-based Query Router Module

This module provides a SLM-based query router using a fine-tuned Phi model
for classifying user queries as either 'local' or 'cloud' based on complexity.

Classes:
    PhiRouterConfig: Configuration for the Phi router
    PhiQueryRouter: Main router class using fine-tuned Phi model
    
Functions:
    analyze_query_characteristics_phi: Analyze query using Phi model
    route_query_phi: Route query using Phi model predictions
"""

import os
import json
import time
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# Hugging Face libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    TextGenerationPipeline
)
from peft import PeftModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PhiRouterConfig:
    """Configuration class for Phi-based query router."""
    model_path: str
    max_length: int = 512
    confidence_threshold: float = 0.7
    device: Optional[str] = None
    batch_size: int = 1
    temperature: float = 0.1
    do_sample: bool = False
    return_full_text: bool = False
    
    def __post_init__(self):
        """Set device automatically if not specified."""
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class PhiQueryRouter:
    """
    Query router using a fine-tuned Phi SLM for intelligent routing decisions.
    
    This router uses a fine-tuned Phi model to classify queries as 'local' or 'cloud'
    based on learned patterns from training data.
    """
    
    def __init__(self, config: PhiRouterConfig):
        """
        Initialize the Phi query router.
        
        Args:
            config: PhiRouterConfig object with model configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.routing_stats = {
            'total_queries': 0,
            'local_routes': 0,
            'cloud_routes': 0,
            'high_confidence_routes': 0,
            'inference_times': []
        }
        
        self._load_model()
        
    def _load_model(self) -> None:
        """Load the fine-tuned Phi model and tokenizer."""
        try:
            logger.info(f"Loading Phi model from: {self.config.model_path}")
            
            # Check if path exists
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model path not found: {self.config.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Set to evaluation mode
            self.model.eval()
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.config.device if self.config.device != "auto" else None,
                torch_dtype=torch_dtype,
                return_full_text=self.config.return_full_text,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                max_new_tokens=10,  # Only need 'local' or 'cloud'
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Phi model loaded successfully")
            logger.info(f"Model device: {next(self.model.parameters()).device}")
            logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
            
        except Exception as e:
            logger.error(f"Failed to load Phi model: {e}")
            raise
    
    def predict(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict the routing destination for a query.
        
        Args:
            query: User query to classify
            
        Returns:
            Tuple of (predicted_label, confidence, scores_dict)
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        start_time = time.time()
        
        try:
            # Format query for Phi model (instruction-following format)
            formatted_prompt = f"<|user|>\n{query}<|end|>\n<|assistant|>\n"
            
            # Generate prediction
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=10,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                return_full_text=self.config.return_full_text,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            if self.config.return_full_text:
                generated_text = outputs[0]['generated_text'].replace(formatted_prompt, "").strip()
            else:
                generated_text = outputs[0]['generated_text'].strip()
            
            # Parse prediction
            prediction, confidence = self._parse_prediction(generated_text, query)
            
            # Create scores dict
            if prediction == 'local':
                scores = {'local': confidence, 'cloud': 1.0 - confidence}
            else:
                scores = {'local': 1.0 - confidence, 'cloud': confidence}
            
            # Update statistics
            inference_time = time.time() - start_time
            self._update_stats(prediction, confidence, inference_time)
            
            return prediction, confidence, scores
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback to simple heuristic
            return self._fallback_prediction(query)
    
    def _parse_prediction(self, generated_text: str, original_query: str) -> Tuple[str, float]:
        """
        Parse the model's generated text to extract prediction and confidence.
        
        Args:
            generated_text: Text generated by the model
            original_query: Original user query
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Clean and normalize the generated text
        cleaned_text = generated_text.lower().strip()
        
        # Look for explicit 'local' or 'cloud' in the response
        if 'local' in cleaned_text and 'cloud' not in cleaned_text:
            prediction = 'local'
            confidence = 0.9  # High confidence for clear response
        elif 'cloud' in cleaned_text and 'local' not in cleaned_text:
            prediction = 'cloud'
            confidence = 0.9  # High confidence for clear response
        elif 'local' in cleaned_text and 'cloud' in cleaned_text:
            # Both mentioned, use position to determine preference
            local_pos = cleaned_text.find('local')
            cloud_pos = cleaned_text.find('cloud')
            if local_pos < cloud_pos:
                prediction = 'local'
            else:
                prediction = 'cloud'
            confidence = 0.7  # Medium confidence for ambiguous response
        else:
            # No clear indication, use fallback heuristic
            return self._heuristic_prediction(original_query)
        
        return prediction, confidence
    
    def _heuristic_prediction(self, query: str) -> Tuple[str, float]:
        """
        Fallback heuristic prediction when model output is unclear.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (prediction, confidence)
        """
        # Simple heuristic based on query characteristics
        word_count = len(query.split())
        
        # Local indicators
        local_keywords = [
            'hello', 'hi', 'hey', 'what is', 'calculate', 'convert',
            'time', 'date', 'simple', 'quick', 'basic'
        ]
        
        # Cloud indicators
        cloud_keywords = [
            'analyze', 'explain', 'describe', 'compare', 'evaluate',
            'comprehensive', 'detailed', 'strategy', 'plan', 'write'
        ]
        
        query_lower = query.lower()
        local_matches = sum(1 for keyword in local_keywords if keyword in query_lower)
        cloud_matches = sum(1 for keyword in cloud_keywords if keyword in query_lower)
        
        if local_matches > cloud_matches or word_count <= 10:
            return 'local', 0.6
        else:
            return 'cloud', 0.6
    
    def _fallback_prediction(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Complete fallback prediction when model fails.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (prediction, confidence, scores)
        """
        prediction, confidence = self._heuristic_prediction(query)
        
        if prediction == 'local':
            scores = {'local': confidence, 'cloud': 1.0 - confidence}
        else:
            scores = {'local': 1.0 - confidence, 'cloud': confidence}
        
        return prediction, confidence, scores
    
    def route_query(self, query: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Route a query and provide detailed reasoning.
        
        Args:
            query: User query to route
            
        Returns:
            Tuple of (target, reason, metadata)
        """
        prediction, confidence, scores = self.predict(query)
        
        # Generate reasoning
        if confidence >= self.config.confidence_threshold:
            confidence_level = "high"
        elif confidence >= 0.6:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        reason = f"Phi model prediction: {prediction} (confidence: {confidence:.3f}, {confidence_level})"
        
        # Create metadata
        metadata = {
            'confidence': confidence,
            'scores': scores,
            'model_used': 'phi_slm',
            'confidence_level': confidence_level,
            'query_length': len(query),
            'word_count': len(query.split())
        }
        
        return prediction, reason, metadata
    
    def batch_predict(self, queries: List[str]) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Predict routing for multiple queries efficiently.
        
        Args:
            queries: List of queries to classify
            
        Returns:
            List of (prediction, confidence, scores) tuples
        """
        results = []
        
        # Process in batches for efficiency
        batch_size = self.config.batch_size
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            
            for query in batch:
                result = self.predict(query)
                results.append(result)
        
        return results
    
    def analyze_query_characteristics(self, query: str) -> Dict[str, Any]:
        """
        Analyze query characteristics using the Phi model.
        
        Args:
            query: Query to analyze
            
        Returns:
            Dictionary with analysis results
        """
        prediction, confidence, scores = self.predict(query)
        
        analysis = {
            'original_query': query,
            'length': len(query),
            'word_count': len(query.split()),
            'phi_prediction': prediction,
            'phi_confidence': confidence,
            'local_score': scores['local'],
            'cloud_score': scores['cloud'],
            'analysis_method': 'phi_slm',
            'model_path': self.config.model_path
        }
        
        return analysis
    
    def _update_stats(self, prediction: str, confidence: float, inference_time: float) -> None:
        """Update internal routing statistics."""
        self.routing_stats['total_queries'] += 1
        
        if prediction == 'local':
            self.routing_stats['local_routes'] += 1
        else:
            self.routing_stats['cloud_routes'] += 1
        
        if confidence >= self.config.confidence_threshold:
            self.routing_stats['high_confidence_routes'] += 1
        
        self.routing_stats['inference_times'].append(inference_time)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive routing statistics."""
        total = self.routing_stats['total_queries']
        
        if total == 0:
            return self.routing_stats
        
        stats = {
            **self.routing_stats,
            'local_percentage': (self.routing_stats['local_routes'] / total) * 100,
            'cloud_percentage': (self.routing_stats['cloud_routes'] / total) * 100,
            'high_confidence_percentage': (self.routing_stats['high_confidence_routes'] / total) * 100,
        }
        
        if self.routing_stats['inference_times']:
            times = self.routing_stats['inference_times']
            stats.update({
                'avg_inference_time': np.mean(times),
                'min_inference_time': np.min(times),
                'max_inference_time': np.max(times),
                'std_inference_time': np.std(times),
                'queries_per_second': 1.0 / np.mean(times) if np.mean(times) > 0 else 0
            })
        
        return stats
    
    def benchmark_inference_speed(self, num_queries: int = 100) -> Dict[str, Any]:
        """
        Benchmark the inference speed of the Phi router.
        
        Args:
            num_queries: Number of test queries to run
            
        Returns:
            Dictionary with benchmark results
        """
        # Sample test queries
        test_queries = [
            "Hello, how are you?",
            "What is 2 + 2?",
            "Analyze the impact of AI on healthcare industry",
            "Convert 100°F to Celsius",
            "Write a comprehensive business plan for a tech startup",
            "What's the capital of France?",
            "Explain machine learning in detail",
            "Good morning!",
            "Compare renewable energy vs fossil fuels",
            "Calculate 15 * 23"
        ]
        
        logger.info(f"Benchmarking Phi router with {num_queries} queries...")
        
        start_time = time.time()
        
        # Run predictions
        for i in range(num_queries):
            query = test_queries[i % len(test_queries)]
            self.predict(query)
        
        total_time = time.time() - start_time
        
        results = {
            'total_queries': num_queries,
            'total_time': total_time,
            'avg_time_per_query': total_time / num_queries,
            'queries_per_second': num_queries / total_time,
            'device': self.config.device,
            'model_path': self.config.model_path
        }
        
        logger.info(f"Benchmark completed: {results['queries_per_second']:.2f} queries/second")
        
        return results


# Backward compatibility functions
def analyze_query_characteristics_phi(query: str, router: PhiQueryRouter) -> Dict[str, Any]:
    """
    Analyze query characteristics using Phi model (backward compatibility).
    
    Args:
        query: Query to analyze
        router: PhiQueryRouter instance
        
    Returns:
        Analysis dictionary
    """
    return router.analyze_query_characteristics(query)


def route_query_phi(query: str, router: PhiQueryRouter) -> Tuple[str, str, Dict[str, Any]]:
    """
    Route query using Phi model (backward compatibility).
    
    Args:
        query: Query to route
        router: PhiQueryRouter instance
        
    Returns:
        Tuple of (target, reason, metadata)
    """
    return router.route_query(query)


# Example usage and testing functions
def create_phi_router(model_path: str, **kwargs) -> PhiQueryRouter:
    """
    Convenience function to create a PhiQueryRouter.
    
    Args:
        model_path: Path to the fine-tuned Phi model
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured PhiQueryRouter instance
    """
    config = PhiRouterConfig(model_path=model_path, **kwargs)
    return PhiQueryRouter(config)


def test_phi_router(router: PhiQueryRouter, test_queries: Optional[List[str]] = None) -> None:
    """
    Test the Phi router with sample queries.
    
    Args:
        router: PhiQueryRouter instance to test
        test_queries: Optional list of test queries
    """
    if test_queries is None:
        test_queries = [
            "Hello there!",
            "What is 15 + 27?",
            "Analyze the impact of artificial intelligence on healthcare",
            "Convert 100°F to Celsius",
            "Write a comprehensive business plan",
            "What's the capital of Japan?",
            "Explain quantum computing in detail",
            "Good morning, how are you?",
            "Compare microservices vs monolithic architecture",
            "Calculate the square root of 144"
        ]
    
    print("🧪 Testing Phi Query Router")
    print("=" * 50)
    
    for query in test_queries:
        target, reason, metadata = router.route_query(query)
        
        print(f"\nQuery: '{query}'")
        print(f"  Route: {target.upper()}")
        print(f"  Confidence: {metadata['confidence']:.3f}")
        print(f"  Reasoning: {reason}")
    
    # Show statistics
    stats = router.get_routing_statistics()
    print(f"\n📊 Router Statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Local routes: {stats.get('local_percentage', 0):.1f}%")
    print(f"  Cloud routes: {stats.get('cloud_percentage', 0):.1f}%")
    print(f"  High confidence: {stats.get('high_confidence_percentage', 0):.1f}%")
    
    if 'avg_inference_time' in stats:
        print(f"  Avg inference time: {stats['avg_inference_time']:.4f}s")
        print(f"  Queries per second: {stats['queries_per_second']:.2f}")


if __name__ == "__main__":
    # Example usage
    print("Phi Query Router Module")
    print("This module provides SLM-based query routing using fine-tuned Phi models.")
    print("\nTo use this module:")
    print("1. Fine-tune a Phi model using finetune_phi_router.py")
    print("2. Create a router: router = create_phi_router('path/to/model')")
    print("3. Route queries: target, reason, metadata = router.route_query('your query')")