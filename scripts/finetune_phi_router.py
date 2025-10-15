#!/usr/bin/env python3
"""
Phi Model Fine-tuning Script for Query Classification

This script fine-tunes a Phi model using LoRA (Low-Rank Adaptation) to classify
user queries as either 'local' or 'cloud' based on complexity and processing requirements.

Based on Microsoft PhiCookBook examples and best practices.

Usage:
    python finetune_phi_router.py --data_file ../data/phi_query_classification_phi_*.jsonl
"""

import argparse
import json
import os
import torch
import wandb
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

# Hugging Face and ML libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import bitsandbytes as bnb
from trl import SFTTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhiRouterFineTuner:
    """
    Fine-tunes Phi models for query classification using LoRA adaptation.
    
    This class handles the complete fine-tuning pipeline from data loading
    to model training and evaluation, specifically optimized for Phi models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the fine-tuner with configuration parameters."""
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_model_and_tokenizer(self) -> None:
        """Load and configure the Phi model and tokenizer."""
        logger.info(f"Loading model: {self.config['model_name']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure model loading arguments
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self.config.get('use_bf16', True) else torch.float16,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True,
        }
        
        # Add quantization if enabled
        if self.config.get('use_quantization', False):
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            logger.info("Using 4-bit quantization")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            **model_kwargs
        )
        
        # Prepare model for training
        if self.config.get('use_quantization', False):
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        logger.info(f"Model loaded successfully. Parameters: {self.model.num_parameters():,}")
        
    def setup_lora(self) -> None:
        """Configure and apply LoRA adaptation to the model."""
        logger.info("Setting up LoRA configuration")
        
        # Define LoRA configuration based on Phi model architecture
        lora_config = LoraConfig(
            r=self.config.get('lora_rank', 16),  # Rank of adaptation
            lora_alpha=self.config.get('lora_alpha', 32),  # LoRA scaling parameter
            target_modules=self.config.get('lora_target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj"  # MLP layers
            ]),
            lora_dropout=self.config.get('lora_dropout', 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"LoRA applied successfully:")
        logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logger.info(f"  Total parameters: {total_params:,}")
        
    def load_and_prepare_data(self) -> None:
        """Load and prepare training and evaluation datasets."""
        logger.info(f"Loading data from: {self.config['data_file']}")
        
        # Load dataset
        if self.config['data_file'].endswith('.jsonl'):
            dataset = load_dataset('json', data_files=self.config['data_file'], split='train')
        else:
            raise ValueError("Data file must be in JSONL format")
        
        logger.info(f"Loaded {len(dataset)} samples")
        
        # Split into train and evaluation sets
        split_ratio = self.config.get('eval_split', 0.1)
        split_dataset = dataset.train_test_split(test_size=split_ratio, seed=42)
        
        self.train_dataset = split_dataset['train']
        self.eval_dataset = split_dataset['test']
        
        logger.info(f"Split dataset:")
        logger.info(f"  Train: {len(self.train_dataset)} samples")
        logger.info(f"  Eval: {len(self.eval_dataset)} samples")
        
        # Tokenize datasets
        self.train_dataset = self.train_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=self.train_dataset.column_names
        )
        
        self.eval_dataset = self.eval_dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=self.eval_dataset.column_names
        )
        
        logger.info("Data tokenization completed")
        
    def _tokenize_function(self, examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        """Tokenize text examples for training."""
        # Tokenize the text
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=self.config.get('max_sequence_length', 512),
            return_tensors=None
        )
        
        # For causal language modeling, labels are the same as input_ids
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized
        
    def setup_training_arguments(self) -> TrainingArguments:
        """Configure training arguments optimized for Phi models."""
        
        # Calculate steps
        total_samples = len(self.train_dataset)
        batch_size = self.config.get('per_device_train_batch_size', 1)
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 4)
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        if torch.cuda.is_available():
            effective_batch_size *= torch.cuda.device_count()
        
        num_epochs = self.config.get('num_train_epochs', 3)
        steps_per_epoch = total_samples // effective_batch_size
        max_steps = steps_per_epoch * num_epochs
        
        # Setup training arguments
        training_args = TrainingArguments(
            # Output and logging
            output_dir=str(self.output_dir),
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=self.config.get('logging_steps', 10),
            logging_strategy="steps",
            
            # Training configuration
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=self.config.get('per_device_eval_batch_size', 1),
            gradient_accumulation_steps=gradient_accumulation_steps,
            
            # Optimization
            learning_rate=self.config.get('learning_rate', 2e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
            lr_scheduler_type=self.config.get('lr_scheduler_type', "cosine"),
            warmup_ratio=self.config.get('warmup_ratio', 0.1),
            
            # Mixed precision and optimization
            fp16=not self.config.get('use_bf16', True),
            bf16=self.config.get('use_bf16', True),
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            
            # Evaluation and saving
            evaluation_strategy="steps",
            eval_steps=self.config.get('eval_steps', 100),
            save_strategy="steps",
            save_steps=self.config.get('save_steps', 100),
            save_total_limit=self.config.get('save_total_limit', 3),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Hardware optimization
            remove_unused_columns=False,
            optim="paged_adamw_32bit" if self.config.get('use_quantization', False) else "adamw_torch",
            
            # Reproducibility
            seed=self.config.get('seed', 42),
            data_seed=self.config.get('seed', 42),
            
            # Reporting
            report_to=self.config.get('report_to', None),
            run_name=self.config.get('run_name', f"phi-router-{datetime.now().strftime('%Y%m%d-%H%M%S')}"),
        )
        
        logger.info(f"Training configuration:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        logger.info(f"  Steps per epoch: {steps_per_epoch}")
        logger.info(f"  Total steps: {max_steps}")
        logger.info(f"  Learning rate: {training_args.learning_rate}")
        
        return training_args
        
    def create_trainer(self, training_args: TrainingArguments) -> SFTTrainer:
        """Create and configure the trainer for supervised fine-tuning."""
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal language modeling
            pad_to_multiple_of=8,
        )
        
        # Setup callbacks
        callbacks = []
        if self.config.get('early_stopping_patience'):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config['early_stopping_patience']
                )
            )
        
        # Create SFT trainer (optimized for instruction following)
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
            max_seq_length=self.config.get('max_sequence_length', 512),
            packing=False,  # Don't pack sequences for classification task
        )
        
        return trainer
        
    def train(self) -> None:
        """Execute the complete training pipeline."""
        logger.info("Starting Phi model fine-tuning pipeline")
        
        # Initialize Weights & Biases if configured
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'phi-query-router'),
                name=self.config.get('run_name'),
                config=self.config
            )
        
        # Load model and setup LoRA
        self.load_model_and_tokenizer()
        self.setup_lora()
        
        # Prepare data
        self.load_and_prepare_data()
        
        # Setup training
        training_args = self.setup_training_arguments()
        trainer = self.create_trainer(training_args)
        
        # Start training
        logger.info("Beginning training...")
        train_result = trainer.train()
        
        # Log training results
        logger.info("Training completed!")
        logger.info(f"Training time: {train_result.metrics['train_runtime']:.2f} seconds")
        logger.info(f"Final training loss: {train_result.metrics['train_loss']:.4f}")
        
        # Save final model and tokenizer
        logger.info("Saving model and tokenizer...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training configuration and results
        config_file = self.output_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        results_file = self.output_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        
        logger.info(f"Model saved to: {self.output_dir}")
        
        # Final evaluation
        if self.eval_dataset:
            logger.info("Running final evaluation...")
            eval_results = trainer.evaluate()
            logger.info(f"Final evaluation loss: {eval_results['eval_loss']:.4f}")
            
            eval_results_file = self.output_dir / "eval_results.json"
            with open(eval_results_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
        
        # Close wandb if used
        if self.config.get('use_wandb', False):
            wandb.finish()
            
        logger.info("Fine-tuning pipeline completed successfully!")


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for Phi model fine-tuning."""
    return {
        # Model configuration
        'model_name': 'microsoft/Phi-3.5-mini-instruct',  # Default Phi model
        'use_quantization': True,  # Use 4-bit quantization for efficiency
        'use_bf16': True,  # Use bfloat16 for mixed precision
        
        # LoRA configuration
        'lora_rank': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'lora_target_modules': [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        
        # Training configuration
        'num_train_epochs': 3,
        'per_device_train_batch_size': 1,
        'per_device_eval_batch_size': 1,
        'gradient_accumulation_steps': 4,
        'learning_rate': 2e-4,
        'weight_decay': 0.01,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.1,
        'max_sequence_length': 512,
        
        # Evaluation and saving
        'eval_split': 0.1,
        'evaluation_strategy': 'steps',
        'eval_steps': 100,
        'save_steps': 100,
        'save_total_limit': 3,
        'logging_steps': 10,
        'early_stopping_patience': 5,
        
        # Output configuration
        'output_dir': './phi_query_router_model',
        'seed': 42,
        
        # Monitoring (optional)
        'use_wandb': False,
        'wandb_project': 'phi-query-router',
        'report_to': None,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Phi model for query classification")
    
    # Data arguments
    parser.add_argument('--data_file', type=str, required=True,
                       help='Path to training data file (JSONL format)')
    parser.add_argument('--output_dir', type=str, default='./phi_query_router_model',
                       help='Output directory for the fine-tuned model')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='microsoft/Phi-3.5-mini',
                       help='Phi model to fine-tune') # Phi-3.5-mini-instruct
    parser.add_argument('--max_sequence_length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Per-device batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--eval_split', type=float, default=0.1,
                       help='Fraction of data to use for evaluation')
    
    # LoRA arguments
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')
    
    # Hardware arguments
    parser.add_argument('--use_quantization', action='store_true',
                       help='Use 4-bit quantization')
    parser.add_argument('--no_bf16', action='store_true',
                       help='Disable bfloat16 (use fp16 instead)')
    
    # Monitoring arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='phi-query-router',
                       help='Weights & Biases project name')
    
    # Configuration file
    parser.add_argument('--config_file', type=str,
                       help='JSON configuration file (overrides command line args)')
    
    args = parser.parse_args()
    
    # Create configuration
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from: {args.config_file}")
    else:
        config = create_default_config()
    
    # Override with command line arguments
    config.update({
        'data_file': args.data_file,
        'output_dir': args.output_dir,
        'model_name': args.model_name,
        'max_sequence_length': args.max_sequence_length,
        'num_train_epochs': args.num_epochs,
        'per_device_train_batch_size': args.batch_size,
        'per_device_eval_batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'eval_split': args.eval_split,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'use_quantization': args.use_quantization,
        'use_bf16': not args.no_bf16,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
    })
    
    # Validate required files
    if not os.path.exists(config['data_file']):
        raise FileNotFoundError(f"Data file not found: {config['data_file']}")
    
    print("🤖 Phi Model Fine-tuning for Query Classification")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Data file: {config['data_file']}")
    print(f"  Output directory: {config['output_dir']}")
    print(f"  Epochs: {config['num_train_epochs']}")
    print(f"  Batch size: {config['per_device_train_batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  LoRA rank: {config['lora_rank']}")
    print(f"  Quantization: {config['use_quantization']}")
    print(f"  Mixed precision: {'bfloat16' if config['use_bf16'] else 'float16'}")
    print()
    
    # Initialize and run training
    fine_tuner = PhiRouterFineTuner(config)
    fine_tuner.train()
    
    print("✅ Fine-tuning completed successfully!")
    print(f"Model saved to: {config['output_dir']}")
    print("\nNext steps:")
    print("1. Test the fine-tuned model with sample queries")
    print("2. Use in the alternate Lab 4 notebook")
    print("3. Evaluate performance compared to rule-based routing")


if __name__ == "__main__":
    main()