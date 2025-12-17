"""
ONNX Converter Module for MobileBERT Query Router

This module provides functionality to convert the trained MobileBERT query router
model to ONNX format for optimized inference across different platforms.

ONNX (Open Neural Network Exchange) enables:
- Faster inference performance
- Cross-platform compatibility
- Hardware acceleration support
- Reduced model size (with quantization)
"""

import os
import json
import torch
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    import onnx
    import onnxruntime as ort
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX dependencies not installed. Run: pip install onnx onnxruntime transformers")


class MobileBertONNXConverter:
    """
    Converts MobileBERT query router models to ONNX format.
    
    Features:
    - PyTorch to ONNX conversion
    - Model validation and testing
    - Performance benchmarking
    - Quantization support
    """
    
    def __init__(self, model_path: str = None, output_dir: str = None):
        """
        Initialize the ONNX converter.
        
        Args:
            model_path: Path to the trained MobileBERT model (uses BERT_MODEL_PATH env var if None)
            output_dir: Directory to save ONNX model (defaults to model_path/onnx)
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX dependencies are required. Install with: pip install onnx onnxruntime")
        
        # Use environment variable if model_path not provided
        if model_path is None:
            model_path = os.getenv('BERT_MODEL_PATH', './notebooks/mobilbert_query_router_trained')
        
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir) if output_dir else self.model_path / "onnx"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.onnx_model_path = None
        self.label_to_id = {"local": 0, "cloud": 1}
        self.id_to_label = {0: "local", 1: "cloud"}
        
        print(f"🔧 MobileBERT ONNX Converter initialized")
        print(f"   Model path: {self.model_path}")
        print(f"   Output directory: {self.output_dir}")
    
    def load_pytorch_model(self):
        """Load the PyTorch model and tokenizer."""
        try:
            print(f"📂 Loading PyTorch model from {self.model_path}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # Load model
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            self.model.eval()
            
            # Load label mappings if available
            config_path = self.model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    model_config = json.load(f)
                    self.label_to_id = model_config.get('label_to_id', self.label_to_id)
                    self.id_to_label = {int(k): v for k, v in model_config.get('id_to_label', self.id_to_label).items()}
            
            print(f"   ✅ PyTorch model and tokenizer loaded successfully")
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"   📊 Model parameters: {total_params:,}")
            
        except Exception as e:
            print(f"   ❌ Failed to load PyTorch model: {e}")
            raise
    
    def convert_to_onnx(self, 
                       opset_version: int = 14,
                       max_length: int = 128,
                       dynamic_axes: bool = True,
                       optimize: bool = True) -> str:
        """
        Convert PyTorch model to ONNX format.
        
        Args:
            opset_version: ONNX opset version (14+ recommended)
            max_length: Maximum sequence length for input
            dynamic_axes: Enable dynamic batch and sequence length
            optimize: Apply ONNX optimization
            
        Returns:
            Path to the exported ONNX model
        """
        if self.model is None:
            self.load_pytorch_model()
        
        print(f"\n🔄 Converting model to ONNX format...")
        print(f"   Opset version: {opset_version}")
        print(f"   Max sequence length: {max_length}")
        print(f"   Dynamic axes: {dynamic_axes}")
        
        # Create dummy input
        dummy_text = "This is a sample query for ONNX conversion"
        inputs = self.tokenizer(
            dummy_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        
        # Define input and output names
        input_names = ['input_ids', 'attention_mask']
        output_names = ['logits']
        
        # Define dynamic axes if enabled
        dynamic_axes_config = None
        if dynamic_axes:
            dynamic_axes_config = {
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size'}
            }
        
        # Export to ONNX
        onnx_model_path = self.output_dir / "mobilebert_query_router.onnx"
        
        try:
            torch.onnx.export(
                self.model,
                (inputs['input_ids'], inputs['attention_mask']),
                str(onnx_model_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_config,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True
            )
            
            print(f"   ✅ Model exported to: {onnx_model_path}")
            self.onnx_model_path = onnx_model_path
            
            # Verify the ONNX model
            print(f"\n🔍 Verifying ONNX model...")
            onnx_model = onnx.load(str(onnx_model_path))
            onnx.checker.check_model(onnx_model)
            print(f"   ✅ ONNX model is valid")
            
            # Get model size
            model_size_mb = onnx_model_path.stat().st_size / (1024 * 1024)
            print(f"   📦 Model size: {model_size_mb:.2f} MB")
            
            # Optimize if requested
            if optimize:
                self._optimize_onnx_model(onnx_model_path)
            
            # Save tokenizer and config
            self._save_tokenizer_and_config(max_length)
            
            return str(onnx_model_path)
            
        except Exception as e:
            print(f"   ❌ ONNX export failed: {e}")
            raise
    
    def _optimize_onnx_model(self, model_path: Path):
        """Apply ONNX Runtime optimizations."""
        try:
            print(f"\n⚡ Optimizing ONNX model...")
            
            from onnxruntime.transformers import optimizer
            from onnxruntime.transformers.fusion_options import FusionOptions
            
            # Configure optimization options
            optimization_options = FusionOptions('bert')
            optimization_options.enable_gelu = True
            optimization_options.enable_layer_norm = True
            optimization_options.enable_attention = True
            optimization_options.enable_skip_layer_norm = True
            optimization_options.enable_embed_layer_norm = True
            optimization_options.enable_bias_skip_layer_norm = True
            optimization_options.enable_bias_gelu = True
            
            # Optimize model
            optimized_model_path = self.output_dir / "mobilebert_query_router_optimized.onnx"
            
            optimizer_instance = optimizer.optimize_model(
                str(model_path),
                model_type='bert',
                num_heads=4,  # MobileBERT typical configuration
                hidden_size=512,
                optimization_options=optimization_options
            )
            
            optimizer_instance.save_model_to_file(str(optimized_model_path))
            
            # Compare sizes
            original_size = model_path.stat().st_size / (1024 * 1024)
            optimized_size = optimized_model_path.stat().st_size / (1024 * 1024)
            reduction = ((original_size - optimized_size) / original_size) * 100
            
            print(f"   ✅ Optimized model saved to: {optimized_model_path}")
            print(f"   📊 Original size: {original_size:.2f} MB")
            print(f"   📊 Optimized size: {optimized_size:.2f} MB")
            print(f"   📉 Size reduction: {reduction:.1f}%")
            
        except Exception as e:
            print(f"   ⚠️  Optimization failed (non-critical): {e}")
    
    def _save_tokenizer_and_config(self, max_length: int):
        """Save tokenizer and configuration for ONNX inference."""
        try:
            # Save tokenizer
            tokenizer_dir = self.output_dir / "tokenizer"
            tokenizer_dir.mkdir(exist_ok=True)
            self.tokenizer.save_pretrained(str(tokenizer_dir))
            print(f"   ✅ Tokenizer saved to: {tokenizer_dir}")
            
            # Save inference configuration
            config = {
                "model_type": "mobilebert",
                "task": "sequence-classification",
                "max_length": max_length,
                "label_to_id": self.label_to_id,
                "id_to_label": self.id_to_label,
                "num_labels": len(self.label_to_id),
                "onnx_opset_version": 14
            }
            
            config_path = self.output_dir / "inference_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            print(f"   ✅ Inference config saved to: {config_path}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to save tokenizer/config: {e}")
    
    def validate_onnx_model(self, test_queries: list = None) -> Dict[str, Any]:
        """
        Validate ONNX model against PyTorch model.
        
        Args:
            test_queries: List of test queries (uses defaults if None)
            
        Returns:
            Validation results dictionary
        """
        if self.onnx_model_path is None:
            raise ValueError("ONNX model not found. Run convert_to_onnx() first.")
        
        if self.model is None:
            self.load_pytorch_model()
        
        print(f"\n🧪 Validating ONNX model...")
        
        # Default test queries
        if test_queries is None:
            test_queries = [
                "Hello there!",
                "What's 5 + 3?",
                "Can you analyze the quarterly sales performance?",
                "Write a comprehensive business plan",
                "Thanks!"
            ]
        
        # Initialize ONNX Runtime session
        ort_session = ort.InferenceSession(str(self.onnx_model_path))
        
        validation_results = {
            'total_tests': len(test_queries),
            'passed': 0,
            'failed': 0,
            'max_difference': 0.0,
            'average_difference': 0.0,
            'test_details': []
        }
        
        differences = []
        
        for query in test_queries:
            # PyTorch inference
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            with torch.no_grad():
                pytorch_outputs = self.model(**inputs)
                pytorch_logits = pytorch_outputs.logits.numpy()
                pytorch_probs = torch.nn.functional.softmax(torch.tensor(pytorch_logits), dim=-1).numpy()
            
            # ONNX inference
            onnx_inputs = {
                'input_ids': inputs['input_ids'].numpy(),
                'attention_mask': inputs['attention_mask'].numpy()
            }
            onnx_outputs = ort_session.run(None, onnx_inputs)
            onnx_logits = onnx_outputs[0]
            onnx_probs = torch.nn.functional.softmax(torch.tensor(onnx_logits), dim=-1).numpy()
            
            # Compare results
            logits_diff = np.abs(pytorch_logits - onnx_logits).max()
            probs_diff = np.abs(pytorch_probs - onnx_probs).max()
            max_diff = max(logits_diff, probs_diff)
            
            differences.append(max_diff)
            
            # Check if predictions match
            pytorch_pred = pytorch_probs.argmax()
            onnx_pred = onnx_probs.argmax()
            predictions_match = pytorch_pred == onnx_pred
            
            test_result = {
                'query': query,
                'pytorch_prediction': self.id_to_label[int(pytorch_pred)],
                'onnx_prediction': self.id_to_label[int(onnx_pred)],
                'predictions_match': bool(predictions_match),
                'max_difference': float(max_diff),
                'pytorch_confidence': float(pytorch_probs.max()),
                'onnx_confidence': float(onnx_probs.max())
            }
            
            validation_results['test_details'].append(test_result)
            
            if predictions_match and max_diff < 1e-4:  # Tolerance threshold
                validation_results['passed'] += 1
            else:
                validation_results['failed'] += 1
            
            print(f"   Query: '{query[:50]}...'")
            print(f"      PyTorch: {test_result['pytorch_prediction']} ({test_result['pytorch_confidence']:.4f})")
            print(f"      ONNX: {test_result['onnx_prediction']} ({test_result['onnx_confidence']:.4f})")
            print(f"      Match: {'✅' if predictions_match else '❌'} (diff: {max_diff:.6f})")
        
        # Calculate summary statistics
        validation_results['max_difference'] = float(max(differences))
        validation_results['average_difference'] = float(np.mean(differences))
        
        print(f"\n📊 Validation Summary:")
        print(f"   Tests passed: {validation_results['passed']}/{validation_results['total_tests']}")
        print(f"   Tests failed: {validation_results['failed']}/{validation_results['total_tests']}")
        print(f"   Max difference: {validation_results['max_difference']:.6f}")
        print(f"   Avg difference: {validation_results['average_difference']:.6f}")
        
        # Save validation results
        results_path = self.output_dir / "validation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2)
        print(f"   ✅ Validation results saved to: {results_path}")
        
        return validation_results
    
    def benchmark_performance(self, num_iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark PyTorch vs ONNX inference performance.
        
        Args:
            num_iterations: Number of iterations for benchmarking
            
        Returns:
            Performance comparison results
        """
        if self.onnx_model_path is None:
            raise ValueError("ONNX model not found. Run convert_to_onnx() first.")
        
        if self.model is None:
            self.load_pytorch_model()
        
        print(f"\n⏱️  Benchmarking performance ({num_iterations} iterations)...")
        
        # Test query
        test_query = "Can you analyze the quarterly business performance and provide recommendations?"
        
        # Prepare inputs
        inputs = self.tokenizer(
            test_query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Benchmark PyTorch
        print(f"\n   Testing PyTorch model...")
        pytorch_times = []
        for i in range(num_iterations):
            start = time.time()
            with torch.no_grad():
                _ = self.model(**inputs)
            pytorch_times.append(time.time() - start)
        
        pytorch_avg = np.mean(pytorch_times)
        pytorch_std = np.std(pytorch_times)
        
        # Benchmark ONNX
        print(f"   Testing ONNX model...")
        ort_session = ort.InferenceSession(str(self.onnx_model_path))
        onnx_inputs = {
            'input_ids': inputs['input_ids'].numpy(),
            'attention_mask': inputs['attention_mask'].numpy()
        }
        
        onnx_times = []
        for i in range(num_iterations):
            start = time.time()
            _ = ort_session.run(None, onnx_inputs)
            onnx_times.append(time.time() - start)
        
        onnx_avg = np.mean(onnx_times)
        onnx_std = np.std(onnx_times)
        
        # Calculate speedup
        speedup = pytorch_avg / onnx_avg if onnx_avg > 0 else 0
        
        results = {
            'num_iterations': num_iterations,
            'pytorch': {
                'avg_time_ms': pytorch_avg * 1000,
                'std_time_ms': pytorch_std * 1000,
                'queries_per_second': 1.0 / pytorch_avg if pytorch_avg > 0 else 0
            },
            'onnx': {
                'avg_time_ms': onnx_avg * 1000,
                'std_time_ms': onnx_std * 1000,
                'queries_per_second': 1.0 / onnx_avg if onnx_avg > 0 else 0
            },
            'speedup': speedup,
            'improvement_percentage': ((pytorch_avg - onnx_avg) / pytorch_avg * 100) if pytorch_avg > 0 else 0
        }
        
        print(f"\n📊 Performance Results:")
        print(f"   PyTorch:")
        print(f"      Average: {results['pytorch']['avg_time_ms']:.2f} ms")
        print(f"      Std Dev: {results['pytorch']['std_time_ms']:.2f} ms")
        print(f"      QPS: {results['pytorch']['queries_per_second']:.2f}")
        print(f"   ONNX:")
        print(f"      Average: {results['onnx']['avg_time_ms']:.2f} ms")
        print(f"      Std Dev: {results['onnx']['std_time_ms']:.2f} ms")
        print(f"      QPS: {results['onnx']['queries_per_second']:.2f}")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Improvement: {results['improvement_percentage']:.1f}%")
        
        # Save benchmark results
        results_path = self.output_dir / "benchmark_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"   ✅ Benchmark results saved to: {results_path}")
        
        return results
    
    def export_for_production(self, 
                            include_optimized: bool = True,
                            include_validation: bool = True,
                            include_benchmark: bool = True) -> Dict[str, str]:
        """
        Complete export pipeline for production deployment.
        
        Args:
            include_optimized: Include optimized ONNX model
            include_validation: Run validation tests
            include_benchmark: Run performance benchmarks
            
        Returns:
            Dictionary with paths to exported artifacts
        """
        print(f"\n🚀 Exporting model for production deployment...")
        
        artifacts = {}
        
        # Convert to ONNX
        onnx_path = self.convert_to_onnx(optimize=include_optimized)
        artifacts['onnx_model'] = onnx_path
        
        # Validation
        if include_validation:
            validation_results = self.validate_onnx_model()
            artifacts['validation_results'] = str(self.output_dir / "validation_results.json")
        
        # Benchmark
        if include_benchmark:
            benchmark_results = self.benchmark_performance()
            artifacts['benchmark_results'] = str(self.output_dir / "benchmark_results.json")
        
        # Create README
        readme_path = self._create_deployment_readme()
        artifacts['readme'] = readme_path
        
        print(f"\n✅ Export completed successfully!")
        print(f"\n📦 Exported Artifacts:")
        for key, path in artifacts.items():
            print(f"   {key}: {path}")
        
        return artifacts
    
    def _create_deployment_readme(self) -> str:
        """Create deployment README file."""
        readme_content = """# MobileBERT Query Router - ONNX Model

## Overview
This directory contains the ONNX-converted MobileBERT query router model optimized for production inference.

## Files
- `mobilebert_query_router.onnx` - Standard ONNX model
- `mobilebert_query_router_optimized.onnx` - Optimized ONNX model (if available)
- `tokenizer/` - Tokenizer files for preprocessing
- `inference_config.json` - Model configuration
- `validation_results.json` - Validation test results
- `benchmark_results.json` - Performance benchmark results

## Usage

### Python with ONNX Runtime
```python
import onnxruntime as ort
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

# Load ONNX model
session = ort.InferenceSession("mobilebert_query_router.onnx")

# Prepare input
query = "Your query here"
inputs = tokenizer(query, return_tensors="np", padding=True, truncation=True, max_length=128)

# Run inference
outputs = session.run(None, {
    'input_ids': inputs['input_ids'],
    'attention_mask': inputs['attention_mask']
})

# Get prediction
logits = outputs[0]
prediction = logits.argmax()
```

## Requirements
```
onnxruntime>=1.16.0
transformers>=4.30.0
numpy>=1.24.0
```

## Performance
See `benchmark_results.json` for detailed performance metrics comparing PyTorch vs ONNX inference.

## Validation
See `validation_results.json` for model validation results ensuring ONNX model matches PyTorch model predictions.
"""
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"   ✅ Deployment README created: {readme_path}")
        return str(readme_path)


# Example usage
if __name__ == "__main__":
    print("🔧 MobileBERT ONNX Converter")
    print("=" * 50)
    
    # Use environment variable for model path
    model_path = os.getenv('BERT_MODEL_PATH', './notebooks/mobilbert_query_router_trained')
    print(f"Using model path: {model_path}")
    
    # Initialize converter
    converter = MobileBertONNXConverter(model_path=model_path)
    
    # Export for production
    artifacts = converter.export_for_production(
        include_optimized=True,
        include_validation=True,
        include_benchmark=True
    )
    
    print("\n✅ Conversion completed successfully!")
