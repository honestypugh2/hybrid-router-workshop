"""
Training Script for mobileBERT-based Query Router

This script trains a mobileBERT model to classify user queries as either
'local' (simple queries) or 'cloud' (complex queries).
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Transformers imports
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict


@dataclass
class ModelConfig:
    """Configuration for the mobileBERT query router."""
    model_name: str = "google/mobilebert-uncased"
    max_length: int = 128
    num_labels: int = 2
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 2
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    logging_steps: int = 50
    seed: int = 42


class QueryRouterTrainer:
    """Trainer class for the mobileBERT query router."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the trainer with configuration."""
        self.config = config
        self.tokenizer = None
        self.model = None
        self.training_args = None
        self.trainer = None
        
        # Set random seeds for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Label mappings
        self.label_to_id = {"local": 0, "cloud": 1}
        self.id_to_label = {0: "local", 1: "cloud"}
        
        print(f"🤖 QueryRouterTrainer initialized with {config.model_name}")
        print(f"   Max length: {config.max_length}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Learning rate: {config.learning_rate}")

    def load_data(self, train_file: str, test_file: str) -> Tuple[Dataset, Dataset]:
        """Load training and test datasets."""
        print(f"\n📂 Loading datasets...")
        
        # Load training data
        with open(train_file, 'r', encoding='utf-8') as f:
            train_json = json.load(f)
        train_data = train_json['data']
        
        # Load test data
        with open(test_file, 'r', encoding='utf-8') as f:
            test_json = json.load(f)
        test_data = test_json['data']
        
        print(f"   Training samples: {len(train_data)}")
        print(f"   Test samples: {len(test_data)}")
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)
        
        return train_dataset, test_dataset

    def setup_model_and_tokenizer(self):
        """Initialize the tokenizer and model."""
        print(f"\n🔧 Setting up tokenizer and model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            id2label=self.id_to_label,
            label2id=self.label_to_id
        )
        
        print(f"   ✅ Tokenizer loaded: {self.config.model_name}")
        print(f"   ✅ Model loaded with {self.config.num_labels} labels")
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   📊 Total parameters: {total_params:,}")
        print(f"   📊 Trainable parameters: {trainable_params:,}")

    def tokenize_function(self, examples):
        """Tokenize the input texts."""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=False,  # Padding will be done by data collator
            max_length=self.config.max_length
        )

    def preprocess_datasets(self, train_dataset: Dataset, test_dataset: Dataset) -> Tuple[Dataset, Dataset]:
        """Preprocess datasets for training."""
        print(f"\n🔄 Preprocessing datasets...")
        
        # Tokenize datasets
        train_tokenized = train_dataset.map(
            self.tokenize_function,
            batched=True,
            desc="Tokenizing train set"
        )
        
        test_tokenized = test_dataset.map(
            self.tokenize_function,
            batched=True,
            desc="Tokenizing test set"
        )
        
        # Set format for PyTorch
        train_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label_id"])
        test_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label_id"])
        
        # Rename label_id to labels (required by Trainer)
        train_tokenized = train_tokenized.rename_column("label_id", "labels")
        test_tokenized = test_tokenized.rename_column("label_id", "labels")
        
        print(f"   ✅ Datasets tokenized and formatted")
        
        return train_tokenized, test_tokenized

    def setup_training_arguments(self, output_dir: str = "./query_router_model"):
        """Setup training arguments."""
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_dir=f"{output_dir}/logs",
            logging_steps=self.config.logging_steps,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            seed=self.config.seed,
            data_seed=self.config.seed,
            report_to=None,  # Disable wandb/tensorboard
            save_total_limit=2,  # Keep only 2 best models
        )
        
        print(f"\n⚙️  Training arguments configured:")
        print(f"   Output directory: {output_dir}")
        print(f"   Learning rate: {self.config.learning_rate}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Epochs: {self.config.num_epochs}")

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train_model(self, train_dataset: Dataset, test_dataset: Dataset):
        """Train the model."""
        print(f"\n🚀 Starting training...")
        
        # Setup data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)]
        )
        
        # Train model
        train_result = self.trainer.train()
        
        print(f"\n✅ Training completed!")
        print(f"   Training time: {train_result.metrics['train_runtime']:.2f} seconds")
        print(f"   Training samples/second: {train_result.metrics['train_samples_per_second']:.2f}")
        print(f"   Final training loss: {train_result.metrics['train_loss']:.4f}")
        
        return train_result

    def evaluate_model(self, test_dataset: Dataset) -> Dict:
        """Evaluate the trained model."""
        print(f"\n📊 Evaluating model...")
        
        # Get predictions
        eval_result = self.trainer.evaluate(eval_dataset=test_dataset)
        
        # Get detailed predictions for confusion matrix
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n📈 Evaluation Results:")
        print(f"   Accuracy: {eval_result['eval_accuracy']:.4f}")
        print(f"   F1 Score: {eval_result['eval_f1']:.4f}")
        print(f"   Precision: {eval_result['eval_precision']:.4f}")
        print(f"   Recall: {eval_result['eval_recall']:.4f}")
        
        # Print confusion matrix
        print(f"\n📋 Confusion Matrix:")
        print(f"                Predicted")
        print(f"Actual     Local   Cloud")
        print(f"Local      {cm[0,0]:5d}   {cm[0,1]:5d}")
        print(f"Cloud      {cm[1,0]:5d}   {cm[1,1]:5d}")
        
        # Calculate per-class metrics
        local_precision = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
        local_recall = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        cloud_precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        cloud_recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        
        print(f"\n🎯 Per-Class Metrics:")
        print(f"   Local - Precision: {local_precision:.4f}, Recall: {local_recall:.4f}")
        print(f"   Cloud - Precision: {cloud_precision:.4f}, Recall: {cloud_recall:.4f}")
        
        # Save confusion matrix plot
        self.plot_confusion_matrix(cm, ["Local", "Cloud"])
        
        return eval_result, cm, y_pred, y_true

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str]):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Query Router - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Confusion matrix saved to: confusion_matrix.png")

    def save_model(self, output_dir: str = "./query_router_model"):
        """Save the trained model and tokenizer."""
        print(f"\n💾 Saving model and tokenizer...")
        
        # Save model and tokenizer
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save configuration
        config_data = {
            "model_name": self.config.model_name,
            "max_length": self.config.max_length,
            "num_labels": self.config.num_labels,
            "label_to_id": self.label_to_id,
            "id_to_label": self.id_to_label,
            "training_config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
                "warmup_ratio": self.config.warmup_ratio,
                "weight_decay": self.config.weight_decay
            },
            "training_date": datetime.now().isoformat(),
            "description": "mobileBERT model trained for query routing (local vs cloud)"
        }
        
        with open(f"{output_dir}/config.json", 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Model saved to: {output_dir}")
        print(f"   ✅ Configuration saved to: {output_dir}/config.json")

    def test_predictions(self, test_queries: List[str]) -> List[Dict]:
        """Test the model with sample queries."""
        print(f"\n🧪 Testing model with sample queries...")
        
        results = []
        
        for query in test_queries:
            # Tokenize query
            inputs = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.max_length
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = predictions.argmax().item()
                confidence = predictions.max().item()
            
            predicted_label = self.id_to_label[predicted_class_id]
            
            result = {
                "query": query,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "local_score": predictions[0][0].item(),
                "cloud_score": predictions[0][1].item()
            }
            results.append(result)
            
            print(f"   Query: '{query}'")
            print(f"   Prediction: {predicted_label.upper()} (confidence: {confidence:.3f})")
            print(f"   Scores - Local: {result['local_score']:.3f}, Cloud: {result['cloud_score']:.3f}")
            print()
        
        return results


def main():
    """Main training function."""
    print("🤖 mobileBERT Query Router Training")
    print("=" * 50)
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configuration
    config = ModelConfig()
    
    # Initialize trainer
    trainer = QueryRouterTrainer(config)
    
    # Load data
    train_dataset, test_dataset = trainer.load_data(
        "query_routing_train.json",
        "query_routing_test.json"
    )
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Preprocess datasets
    train_dataset, test_dataset = trainer.preprocess_datasets(train_dataset, test_dataset)
    
    # Setup training arguments
    trainer.setup_training_arguments("./mobilebert_query_router")
    
    # Train model
    train_result = trainer.train_model(train_dataset, test_dataset)
    
    # Evaluate model
    eval_result, cm, y_pred, y_true = trainer.evaluate_model(test_dataset)
    
    # Save model
    trainer.save_model("./mobilebert_query_router")
    
    # Test with sample queries
    test_queries = [
        "Hello there!",
        "What is 25 + 17?",
        "Analyze the impact of artificial intelligence on healthcare",
        "What's the capital of France?",
        "Write a comprehensive business plan for a fintech startup",
        "Convert 100 degrees Fahrenheit to Celsius",
        "Explain the methodology for implementing agile software development"
    ]
    
    results = trainer.test_predictions(test_queries)
    
    # Save test results
    with open("model_test_results.json", 'w', encoding='utf-8') as f:
        json.dump({
            "test_queries": results,
            "evaluation_metrics": eval_result,
            "training_config": config.__dict__,
            "test_date": datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n🎉 Training completed successfully!")
    print(f"   Model saved to: ./mobilebert_query_router")
    print(f"   Test results saved to: model_test_results.json")
    print(f"   Final accuracy: {eval_result['eval_accuracy']:.4f}")


if __name__ == "__main__":
    main()