"""
BERT-based Query Router Module

This module implements a BERT-based query router using a trained mobileBERT model
to classify user queries as either 'local' (simple queries) or 'cloud' (complex queries).
"""

import os
import json
import torch
import time
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from dotenv import load_dotenv

load_dotenv()


@dataclass
class BertRouterConfig:
    """Configuration for the BERT-based router."""
    model_path: str = os.environ["BERT_MODEL_PATH"]
    max_length: int = int(os.environ["BERT_MAX_SEQUENCE_LENGTH"])
    confidence_threshold: float = float(os.environ["BERT_CONFIDENCE_THRESHOLD"])
    device: Optional[str] = None


class BertQueryRouter:
    """
    BERT-based query router that uses a trained mobileBERT model
    to intelligently route queries between local and cloud models.
    """
    
    def __init__(self, config: BertRouterConfig):
        """
        Initialize the BERT router.
        
        Args:
            config: Configuration for the router
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.label_to_id = {"local": 0, "cloud": 1}
        self.id_to_label = {0: "local", 1: "cloud"}
        
        # Set device
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        print(f"🤖 BertQueryRouter initialized")
        print(f"   Model path: {config.model_path}")
        print(f"   Device: {self.device}")
        print(f"   Max length: {config.max_length}")
        print(f"   Confidence threshold: {config.confidence_threshold}")
        
        # Load model and tokenizer
        self._load_model()
        
        # Initialize statistics
        self.routing_stats = {
            'total_queries': 0,
            'local_routes': 0,
            'cloud_routes': 0,
            'high_confidence_predictions': 0,
            'low_confidence_predictions': 0,
            'total_inference_time': 0.0,
            'average_inference_time': 0.0
        }

    def _load_model(self):
        """Load the trained BERT model and tokenizer."""
        try:
            print(f"📂 Loading model from {self.config.model_path}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load configuration if available
            config_path = os.path.join(self.config.model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    model_config = json.load(f)
                    self.label_to_id = model_config.get('label_to_id', self.label_to_id)
                    self.id_to_label = model_config.get('id_to_label', self.id_to_label)
                    print(f"   ✅ Model configuration loaded")
            
            print(f"   ✅ Model and tokenizer loaded successfully")
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"   📊 Model parameters: {total_params:,}")
            
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            raise

    def predict(self, query: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict routing decision for a query.
        
        Args:
            query: Input query string
            
        Returns:
            Tuple of (predicted_label, confidence, scores_dict)
        """
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.max_length
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.inference_mode():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = predictions.argmax().item()
                confidence = predictions.max().item()
            
            # Convert to CPU and extract scores
            predictions_cpu = predictions.cpu().numpy()[0]
            local_score = float(predictions_cpu[0])
            cloud_score = float(predictions_cpu[1])
            
            predicted_label = self.id_to_label[predicted_class_id]
            
            scores = {
                'local': local_score,
                'cloud': cloud_score
            }
            
            # Update statistics
            inference_time = time.time() - start_time
            self._update_stats(predicted_label, confidence, inference_time)
            
            return predicted_label, confidence, scores
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Fallback to simple heuristic
            word_count = len(query.split())
            if word_count <= 10:
                return "local", 0.5, {"local": 0.6, "cloud": 0.4}
            else:
                return "cloud", 0.5, {"local": 0.4, "cloud": 0.6}

    def route_query(self, query: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        Route a query with detailed reasoning.
        
        Args:
            query: Input query string
            
        Returns:
            Tuple of (target_model, reasoning, metadata)
        """
        # Get BERT prediction
        predicted_label, confidence, scores = self.predict(query)
        
        # Create detailed reasoning
        confidence_level = "high" if confidence >= self.config.confidence_threshold else "low"
        
        reasoning = (f"BERT model prediction: {predicted_label} "
                    f"(confidence: {confidence:.3f}, {confidence_level})")
        
        # Add confidence qualifier to reasoning
        if confidence < self.config.confidence_threshold:
            reasoning += f" - Low confidence prediction, consider manual review"
        
        # Metadata for analysis
        metadata = {
            'prediction_method': 'bert',
            'predicted_label': predicted_label,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'scores': scores,
            'query_length': len(query),
            'word_count': len(query.split()),
            'meets_confidence_threshold': confidence >= self.config.confidence_threshold
        }
        
        return predicted_label, reasoning, metadata

    def analyze_query_characteristics(self, query: str) -> Dict[str, Any]:
        """
        Analyze query characteristics using BERT.
        
        Args:
            query: Input query string
            
        Returns:
            Dictionary with analysis results
        """
        predicted_label, confidence, scores = self.predict(query)
        
        return {
            'original_query': query,
            'length': len(query),
            'word_count': len(query.split()),
            'bert_prediction': predicted_label,
            'bert_confidence': confidence,
            'local_score': scores['local'],
            'cloud_score': scores['cloud'],
            'score_difference': abs(scores['local'] - scores['cloud']),
            'high_confidence': confidence >= self.config.confidence_threshold,
            'analysis_method': 'bert_mobilbert'
        }

    def batch_predict(self, queries: list) -> list:
        """
        Predict routing for multiple queries efficiently.
        
        Args:
            queries: List of query strings
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"🔄 Processing {len(queries)} queries in batch...")
        
        for i, query in enumerate(queries):
            if i % 100 == 0 and i > 0:
                print(f"   Processed {i}/{len(queries)} queries...")
            
            predicted_label, confidence, scores = self.predict(query)
            
            results.append({
                'query': query,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'scores': scores
            })
        
        print(f"   ✅ Batch processing completed")
        return results

    def _update_stats(self, predicted_label: str, confidence: float, inference_time: float):
        """Update routing statistics."""
        self.routing_stats['total_queries'] += 1
        self.routing_stats['total_inference_time'] += inference_time
        
        if predicted_label == 'local':
            self.routing_stats['local_routes'] += 1
        else:
            self.routing_stats['cloud_routes'] += 1
        
        if confidence >= self.config.confidence_threshold:
            self.routing_stats['high_confidence_predictions'] += 1
        else:
            self.routing_stats['low_confidence_predictions'] += 1
        
        # Update average inference time
        total_queries = self.routing_stats['total_queries']
        self.routing_stats['average_inference_time'] = (
            self.routing_stats['total_inference_time'] / total_queries
        )

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get detailed routing statistics."""
        total = self.routing_stats['total_queries']
        if total == 0:
            return self.routing_stats.copy()
        
        stats = self.routing_stats.copy()
        stats.update({
            'local_percentage': (self.routing_stats['local_routes'] / total) * 100,
            'cloud_percentage': (self.routing_stats['cloud_routes'] / total) * 100,
            'high_confidence_percentage': (self.routing_stats['high_confidence_predictions'] / total) * 100,
            'low_confidence_percentage': (self.routing_stats['low_confidence_predictions'] / total) * 100
        })
        
        return stats

    def reset_statistics(self):
        """Reset all routing statistics."""
        self.routing_stats = {
            'total_queries': 0,
            'local_routes': 0,
            'cloud_routes': 0,
            'high_confidence_predictions': 0,
            'low_confidence_predictions': 0,
            'total_inference_time': 0.0,
            'average_inference_time': 0.0
        }

    def save_statistics(self, filename: str = None):
        """Save routing statistics to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bert_router_stats_{timestamp}.json"
        
        stats_data = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model_path': self.config.model_path,
                'max_length': self.config.max_length,
                'confidence_threshold': self.config.confidence_threshold,
                'device': str(self.device)
            },
            'statistics': self.get_routing_statistics()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Statistics saved to: {filename}")
        return filename

    def adjust_confidence_threshold(self, new_threshold: float):
        """
        Adjust the confidence threshold for routing decisions.
        
        Args:
            new_threshold: New threshold value (0.0 to 1.0)
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        old_threshold = self.config.confidence_threshold
        self.config.confidence_threshold = new_threshold
        
        print(f"🔧 Confidence threshold adjusted: {old_threshold:.3f} → {new_threshold:.3f}")

    def benchmark_inference_speed(self, num_queries: int = 100) -> Dict[str, float]:
        """
        Benchmark inference speed with sample queries.
        
        Args:
            num_queries: Number of queries to test
            
        Returns:
            Dictionary with benchmark results
        """
        print(f"⏱️  Benchmarking inference speed with {num_queries} queries...")
        
        # Generate sample queries
        sample_queries = [
            f"Sample query number {i} for benchmarking inference speed"
            for i in range(num_queries)
        ]
        
        # Time the predictions
        start_time = time.time()
        for query in sample_queries:
            self.predict(query)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_query = total_time / num_queries
        queries_per_second = num_queries / total_time
        
        benchmark_results = {
            'num_queries': num_queries,
            'total_time': total_time,
            'avg_time_per_query': avg_time_per_query,
            'queries_per_second': queries_per_second,
            'device': str(self.device)
        }
        
        print(f"   ✅ Benchmark completed:")
        print(f"   Total time: {total_time:.3f}s")
        print(f"   Average time per query: {avg_time_per_query:.4f}s")
        print(f"   Queries per second: {queries_per_second:.2f}")
        
        return benchmark_results


# Convenience functions for backward compatibility with rule-based router
def analyze_query_characteristics(query: str, router: BertQueryRouter = None) -> Dict[str, Any]:
    """Analyze query characteristics using BERT router."""
    if router is None:
        # Create default router if none provided
        config = BertRouterConfig()
        router = BertQueryRouter(config)
    
    return router.analyze_query_characteristics(query)


def route_query(query: str, router: BertQueryRouter = None) -> Tuple[str, str]:
    """Route query using BERT router (returns string target for compatibility)."""
    if router is None:
        # Create default router if none provided
        config = BertRouterConfig()
        router = BertQueryRouter(config)
    
    target, reason, _ = router.route_query(query)
    return target, reason


# Example usage and testing
if __name__ == "__main__":
    # Test the BERT router
    print("🤖 BERT Query Router Test")
    print("=" * 30)
    
    # Initialize router
    config = BertRouterConfig(model_path=os.environ["BERT_MODEL_PATH"])
    router = BertQueryRouter(config)
    
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
    
    print(f"\n🧪 Testing BERT Router:")
    print("=" * 40)
    
    for query in test_queries:
        target, reason, metadata = router.route_query(query)
        
        print(f"\n📝 Query: {query}")
        print(f"🎯 Target: {target.upper()}")
        print(f"📊 Confidence: {metadata['confidence']:.3f}")
        print(f"💭 Reason: {reason}")
    
    # Show statistics
    stats = router.get_routing_statistics()
    print(f"\n📈 Routing Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Benchmark speed
    benchmark_results = router.benchmark_inference_speed(50)
    
    print(f"\n✅ BERT router test completed!")