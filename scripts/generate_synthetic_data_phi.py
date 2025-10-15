#!/usr/bin/env python3
"""
Synthetic Data Generator for Phi-based Query Router Fine-tuning

This script generates synthetic training data for fine-tuning a Phi model to classify
user queries as either 'local' or 'cloud' based on complexity and processing requirements.

Usage:
    python generate_synthetic_data_phi.py --num_samples 4000 --output_dir ../data
"""

import argparse
import json
import random
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

class PhiSyntheticDataGenerator:
    """
    Generates synthetic data for training a Phi-based query classification model.
    
    The generator creates realistic user queries categorized as either:
    - 'local': Simple queries suitable for local/edge processing
    - 'cloud': Complex queries requiring cloud-based processing
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize the generator with predefined templates and patterns."""
        random.seed(random_seed)
        
        # Local query templates (simple, fast responses)
        self.local_templates = {
            'greetings': [
                "Hello",
                "Hi there",
                "Good morning",
                "Hey",
                "How are you?",
                "Good afternoon",
                "Hello, how are you doing?",
                "Hi, what's up?",
                "Good evening",
                "Greetings"
            ],
            'simple_calculations': [
                "What is {a} + {b}?",
                "Calculate {a} * {b}",
                "What's {a} minus {b}?",
                "Divide {a} by {b}",
                "What is {a} percent of {b}?",
                "Convert {temp}°F to Celsius",
                "Convert {temp}°C to Fahrenheit",
                "What's the square root of {num}?",
                "Calculate {a} to the power of {b}",
                "What is {a} divided by {b}?"
            ],
            'basic_facts': [
                "What is the capital of {country}?",
                "Who invented the {invention}?",
                "When was {thing} created?",
                "What does {acronym} stand for?",
                "How many {unit} in a {larger_unit}?",
                "What is the population of {city}?",
                "What year was {person} born?",
                "What is the {element} symbol?",
                "Who wrote {book}?",
                "What is the distance from {city1} to {city2}?"
            ],
            'simple_definitions': [
                "Define {term}",
                "What is {concept}?",
                "Explain {simple_concept} briefly",
                "What does {word} mean?",
                "Give me a quick definition of {term}",
                "What is the meaning of {phrase}?",
                "Define the term {technical_term}",
                "What is {basic_concept}?",
                "Briefly explain {simple_idea}",
                "What does the word {vocabulary} mean?"
            ],
            'time_date': [
                "What time is it?",
                "What day is today?",
                "What's the date?",
                "What month is it?",
                "What year is it?",
                "What day of the week is it?",
                "Tell me the current time",
                "What's today's date?",
                "Give me the time",
                "What's the current date and time?"
            ],
            'simple_conversions': [
                "Convert {amount} {unit1} to {unit2}",
                "How many {small_unit} are in {amount} {large_unit}?",
                "Change {value} {from_unit} to {to_unit}",
                "What is {measurement} in {new_unit}?",
                "Convert {quantity} from {old_unit} to {new_unit}",
                "How much is {amount} {currency1} in {currency2}?",
                "Transform {value} {metric} to {imperial}",
                "Change {number} {original} into {target}",
                "What's {amount} {unit} converted to {other_unit}?",
                "Turn {value} {from_format} into {to_format}"
            ]
        }
        
        # Cloud query templates (complex, requiring sophisticated processing)
        self.cloud_templates = {
            'analysis_requests': [
                "Analyze the impact of {topic} on {domain}",
                "Provide a comprehensive analysis of {subject}",
                "Evaluate the effectiveness of {strategy} in {context}",
                "Compare and contrast {concept1} versus {concept2}",
                "Assess the implications of {event} for {industry}",
                "Examine the relationship between {factor1} and {factor2}",
                "Investigate the causes and effects of {phenomenon}",
                "Analyze the trends in {field} over the past {timeframe}",
                "Evaluate the pros and cons of {approach} in {scenario}",
                "Study the correlation between {variable1} and {variable2}"
            ],
            'creative_writing': [
                "Write a {type} about {theme}",
                "Create a {length} story featuring {character}",
                "Compose a poem about {subject}",
                "Draft a {document_type} for {purpose}",
                "Generate creative content about {topic}",
                "Write a script for {scenario}",
                "Create a narrative describing {situation}",
                "Compose lyrics for a song about {theme}",
                "Draft a creative piece on {subject}",
                "Write an imaginative story involving {elements}"
            ],
            'strategic_planning': [
                "Develop a {plan_type} for {organization}",
                "Create a comprehensive strategy for {goal}",
                "Design a roadmap for {objective}",
                "Formulate a plan to achieve {target}",
                "Outline a strategy for {challenge}",
                "Develop a framework for {initiative}",
                "Create an action plan for {project}",
                "Design a methodology for {process}",
                "Formulate an approach to {problem}",
                "Outline a comprehensive plan for {venture}"
            ],
            'detailed_explanations': [
                "Explain in detail how {process} works",
                "Provide a comprehensive overview of {system}",
                "Describe the complete process of {procedure}",
                "Give a thorough explanation of {concept}",
                "Walk me through the entire {workflow}",
                "Explain the underlying principles of {technology}",
                "Provide an in-depth analysis of {mechanism}",
                "Describe the step-by-step process of {operation}",
                "Give a detailed breakdown of {methodology}",
                "Explain the complex relationships in {domain}"
            ],
            'research_synthesis': [
                "Summarize the latest research on {field}",
                "Synthesize findings from multiple studies about {topic}",
                "Review the current state of {research_area}",
                "Compile recent developments in {domain}",
                "Aggregate research findings on {subject}",
                "Survey the literature on {academic_topic}",
                "Synthesize expert opinions on {controversial_topic}",
                "Review and analyze studies related to {phenomenon}",
                "Compile a comprehensive review of {research_field}",
                "Summarize interdisciplinary research on {complex_topic}"
            ],
            'problem_solving': [
                "How can we solve the problem of {issue} in {context}?",
                "What are potential solutions to {challenge}?",
                "Recommend approaches to address {problem}",
                "Suggest strategies for overcoming {obstacle}",
                "How might we resolve {conflict} in {situation}?",
                "What are innovative solutions for {complex_issue}?",
                "Propose methods to tackle {systematic_problem}",
                "How can organizations address {widespread_issue}?",
                "What are effective ways to handle {difficult_situation}?",
                "Recommend solutions for {multifaceted_problem}"
            ]
        }
        
        # Placeholder values for template substitution
        self.placeholders = {
            'a': [str(i) for i in range(1, 101)],
            'b': [str(i) for i in range(1, 51)],
            'num': [str(i) for i in [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144]],
            'temp': [str(i) for i in range(32, 101, 5)],
            'country': ['France', 'Germany', 'Japan', 'Australia', 'Canada', 'Brazil', 'India', 'China'],
            'invention': ['telephone', 'computer', 'airplane', 'internet', 'television', 'radio'],
            'thing': ['Python', 'JavaScript', 'the iPhone', 'Facebook', 'Google', 'Microsoft'],
            'acronym': ['API', 'CPU', 'GPU', 'RAM', 'SSD', 'URL', 'HTTP', 'JSON'],
            'unit': ['meters', 'feet', 'kilograms', 'pounds', 'liters', 'gallons'],
            'larger_unit': ['kilometer', 'mile', 'ton', 'gallon'],
            'city': ['Paris', 'Tokyo', 'New York', 'London', 'Sydney', 'Toronto'],
            'person': ['Einstein', 'Shakespeare', 'Lincoln', 'Mozart', 'Da Vinci'],
            'element': ['gold', 'silver', 'oxygen', 'hydrogen', 'carbon'],
            'book': ['1984', 'Pride and Prejudice', 'The Great Gatsby', 'To Kill a Mockingbird'],
            'city1': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
            'city2': ['Miami', 'Seattle', 'Boston', 'Denver'],
            'topic': ['artificial intelligence', 'climate change', 'renewable energy', 'automation', 'globalization'],
            'domain': ['healthcare', 'education', 'finance', 'manufacturing', 'transportation'],
            'subject': ['machine learning', 'blockchain technology', 'cybersecurity', 'data analytics'],
            'strategy': ['agile methodology', 'digital transformation', 'lean manufacturing', 'remote work'],
            'context': ['small businesses', 'large enterprises', 'startups', 'government agencies'],
            'concept1': ['microservices', 'monolithic architecture', 'cloud computing', 'edge computing'],
            'concept2': ['traditional systems', 'legacy infrastructure', 'on-premise solutions'],
            'type': ['short story', 'poem', 'essay', 'screenplay', 'novel'],
            'theme': ['technology', 'nature', 'friendship', 'adventure', 'mystery'],
            'length': ['brief', 'detailed', 'comprehensive', 'concise'],
            'character': ['a robot', 'a scientist', 'an explorer', 'a teacher'],
            'plan_type': ['business plan', 'marketing strategy', 'growth plan', 'expansion strategy'],
            'organization': ['a tech startup', 'a non-profit', 'a healthcare system', 'a university'],
            'process': ['photosynthesis', 'machine learning', 'blockchain validation', 'protein synthesis'],
            'field': ['quantum computing', 'genetic engineering', 'renewable energy', 'space exploration']
        }
    
    def _substitute_placeholders(self, template: str) -> str:
        """Substitute placeholders in templates with random values."""
        result = template
        for placeholder, values in self.placeholders.items():
            if '{' + placeholder + '}' in result:
                result = result.replace('{' + placeholder + '}', random.choice(values))
        return result
    
    def generate_local_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate samples for local processing (simple queries)."""
        samples = []
        categories = list(self.local_templates.keys())
        
        for _ in range(num_samples):
            category = random.choice(categories)
            template = random.choice(self.local_templates[category])
            query = self._substitute_placeholders(template)
            
            # Format for Phi fine-tuning (instruction-following format)
            formatted_text = f"<|user|>\n{query}<|end|>\n<|assistant|>\nlocal<|end|>"
            
            sample = {
                'text': formatted_text,
                'query': query,
                'label': 'local',
                'category': category,
                'confidence': round(random.uniform(0.85, 0.99), 3),
                'reasoning': f"Simple {category.replace('_', ' ')} query suitable for fast local processing"
            }
            samples.append(sample)
        
        return samples
    
    def generate_cloud_samples(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate samples for cloud processing (complex queries)."""
        samples = []
        categories = list(self.cloud_templates.keys())
        
        for _ in range(num_samples):
            category = random.choice(categories)
            template = random.choice(self.cloud_templates[category])
            query = self._substitute_placeholders(template)
            
            # Add complexity to some queries
            if random.random() < 0.3:  # 30% chance to add complexity
                complexity_additions = [
                    " with detailed examples and case studies",
                    " considering multiple perspectives and stakeholder views",
                    " including potential risks and mitigation strategies",
                    " with comprehensive implementation roadmap",
                    " incorporating latest industry best practices"
                ]
                query += random.choice(complexity_additions)
            
            # Format for Phi fine-tuning (instruction-following format)
            formatted_text = f"<|user|>\n{query}<|end|>\n<|assistant|>\ncloud<|end|>"
            
            sample = {
                'text': formatted_text,
                'query': query,
                'label': 'cloud',
                'category': category,
                'confidence': round(random.uniform(0.80, 0.95), 3),
                'reasoning': f"Complex {category.replace('_', ' ')} requiring sophisticated cloud processing"
            }
            samples.append(sample)
        
        return samples
    
    def generate_edge_cases(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate edge cases and ambiguous queries for robust training."""
        edge_cases = []
        
        # Medium complexity queries (could go either way)
        medium_complexity_templates = [
            "How does {process} work?",
            "What are the benefits of {technology}?",
            "Explain {concept} to me",
            "What is the difference between {term1} and {term2}?",
            "Why is {topic} important?",
            "How do I {action}?",
            "What are the main types of {category}?",
            "When should I use {tool}?",
            "What makes {thing} effective?",
            "How can {approach} help with {problem}?"
        ]
        
        # Generate medium complexity samples
        for _ in range(num_samples // 2):
            template = random.choice(medium_complexity_templates)
            query = self._substitute_placeholders(template)
            
            # Randomly assign to local or cloud (50/50 for edge cases)
            label = random.choice(['local', 'cloud'])
            confidence = round(random.uniform(0.55, 0.75), 3)  # Lower confidence for edge cases
            
            formatted_text = f"<|user|>\n{query}<|end|>\n<|assistant|>\n{label}<|end|>"
            
            edge_case = {
                'text': formatted_text,
                'query': query,
                'label': label,
                'category': 'edge_case_medium',
                'confidence': confidence,
                'reasoning': f"Medium complexity query - edge case with {label} assignment"
            }
            edge_cases.append(edge_case)
        
        # Generate contextual variations (same query, different contexts)
        base_queries = [
            "How does this work?",
            "What should I do?",
            "Can you help me?",
            "What's the best approach?",
            "How do I solve this problem?"
        ]
        
        for _ in range(num_samples // 2):
            base_query = random.choice(base_queries)
            
            # Add context that influences routing
            if random.random() < 0.5:  # Local context
                contexts = [
                    " with basic information",
                    " quickly",
                    " in simple terms",
                    " with a short answer"
                ]
                query = base_query + random.choice(contexts)
                label = 'local'
            else:  # Cloud context
                contexts = [
                    " with detailed analysis",
                    " considering all factors",
                    " with comprehensive examples",
                    " including pros and cons"
                ]
                query = base_query + random.choice(contexts)
                label = 'cloud'
            
            confidence = round(random.uniform(0.60, 0.80), 3)
            formatted_text = f"<|user|>\n{query}<|end|>\n<|assistant|>\n{label}<|end|>"
            
            edge_case = {
                'text': formatted_text,
                'query': query,
                'label': label,
                'category': 'edge_case_contextual',
                'confidence': confidence,
                'reasoning': f"Contextual query requiring {label} processing based on complexity indicators"
            }
            edge_cases.append(edge_case)
        
        return edge_cases
    
    def generate_dataset(self, total_samples: int, local_ratio: float = 0.45) -> List[Dict[str, Any]]:
        """
        Generate a complete dataset with specified distribution.
        
        Args:
            total_samples: Total number of samples to generate
            local_ratio: Proportion of samples for local processing (default: 45%)
        
        Returns:
            List of samples ready for training
        """
        local_samples = int(total_samples * local_ratio)
        cloud_samples = int(total_samples * (1 - local_ratio) * 0.9)  # 90% of remaining for cloud
        edge_samples = total_samples - local_samples - cloud_samples  # Remaining for edge cases
        
        print(f"Generating {total_samples} samples:")
        print(f"  Local: {local_samples} ({local_samples/total_samples*100:.1f}%)")
        print(f"  Cloud: {cloud_samples} ({cloud_samples/total_samples*100:.1f}%)")
        print(f"  Edge cases: {edge_samples} ({edge_samples/total_samples*100:.1f}%)")
        
        dataset = []
        
        # Generate samples
        print("Generating local samples...")
        dataset.extend(self.generate_local_samples(local_samples))
        
        print("Generating cloud samples...")
        dataset.extend(self.generate_cloud_samples(cloud_samples))
        
        print("Generating edge cases...")
        dataset.extend(self.generate_edge_cases(edge_samples))
        
        # Shuffle the dataset
        random.shuffle(dataset)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], output_dir: str, filename_prefix: str = "phi_query_classification"):
        """Save dataset in multiple formats for different use cases."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save for Phi fine-tuning (JSONL format)
        jsonl_file = output_path / f"{filename_prefix}_phi_{timestamp}.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for sample in dataset:
                # Save only the text field for training
                json.dump({'text': sample['text']}, f, ensure_ascii=False)
                f.write('\n')
        
        # Save complete dataset with metadata (JSON format)
        json_file = output_path / f"{filename_prefix}_complete_{timestamp}.json"
        
        # Validate dataset is JSON serializable
        try:
            json.dumps(dataset)
        except TypeError as e:
            print(f"Warning: Dataset contains non-JSON serializable data: {e}")
            print("Applying data cleaning...")
            dataset = self._ensure_json_serializable(dataset)
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for analysis
        csv_file = output_path / f"{filename_prefix}_analysis_{timestamp}.csv"
        df = pd.DataFrame(dataset)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Generate dataset statistics
        stats = self.generate_statistics(dataset)
        
        # Validate that stats are JSON serializable before writing
        try:
            json.dumps(stats)
        except TypeError as e:
            print(f"Warning: Statistics contain non-JSON serializable data: {e}")
            print("Applying additional data cleaning...")
            stats = self._ensure_json_serializable(stats)
        
        stats_file = output_path / f"{filename_prefix}_stats_{timestamp}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\nDataset saved:")
        print(f"  Training file (JSONL): {jsonl_file}")
        print(f"  Complete data (JSON): {json_file}")
        print(f"  Analysis file (CSV): {csv_file}")
        print(f"  Statistics file: {stats_file}")
        
        return {
            'training_file': str(jsonl_file),
            'complete_file': str(json_file),
            'analysis_file': str(csv_file),
            'stats_file': str(stats_file)
        }
    
    def _ensure_json_serializable(self, data: Any) -> Any:
        """Ensure data is JSON serializable by converting problematic types."""
        if isinstance(data, dict):
            # Convert any tuple keys to strings and recursively clean values
            cleaned = {}
            for key, value in data.items():
                if isinstance(key, tuple):
                    # Convert tuple keys to string representation
                    key = str(key)
                elif not isinstance(key, (str, int, float, bool, type(None))):
                    key = str(key)
                cleaned[key] = self._ensure_json_serializable(value)
            return cleaned
        elif isinstance(data, (list, tuple)):
            return [self._ensure_json_serializable(item) for item in data]
        elif isinstance(data, (pd.Series, pd.DataFrame)):
            return self._ensure_json_serializable(data.to_dict())
        elif hasattr(data, 'item'):  # numpy scalars
            return data.item()
        elif isinstance(data, (int, float, str, bool, type(None))):
            return data
        else:
            # Convert any other type to string as fallback
            return str(data)
    
    def generate_statistics(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive statistics about the dataset."""
        df = pd.DataFrame(dataset)
        
        stats = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(dataset),
                'generator_version': '1.0.0'
            },
            'label_distribution': df['label'].value_counts().to_dict(),
            'category_distribution': df['category'].value_counts().to_dict(),
            'confidence_stats': {
                'mean': float(df['confidence'].mean()),
                'std': float(df['confidence'].std()),
                'min': float(df['confidence'].min()),
                'max': float(df['confidence'].max()),
                'by_label': df.groupby('label')['confidence'].apply(lambda x: {
                    'mean': float(x.mean()),
                    'std': float(x.std())
                }).to_dict()
            },
            'query_length_stats': {
                'mean_chars': float(df['query'].str.len().mean()),
                'mean_words': float(df['query'].str.split().str.len().mean()),
                'by_label': df.groupby('label')['query'].apply(lambda x: {
                    'mean_chars': float(x.str.len().mean()),
                    'mean_words': float(x.str.split().str.len().mean())
                }).to_dict()
            },
            'text_length_stats': {
                'mean_chars': float(df['text'].str.len().mean()),
                'by_label': df.groupby('label')['text'].apply(lambda x: {
                    'mean_chars': float(x.str.len().mean()),
                    'std_chars': float(x.str.len().std())
                }).to_dict()
            }
        }
        
        # Ensure all data is JSON serializable
        stats = self._ensure_json_serializable(stats)
        
        return stats


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for Phi-based query router")
    parser.add_argument('--num_samples', type=int, default=4000, 
                       help='Total number of samples to generate (default: 4000)')
    parser.add_argument('--output_dir', type=str, default='../data',
                       help='Output directory for generated data (default: ../data)')
    parser.add_argument('--local_ratio', type=float, default=0.45,
                       help='Proportion of local samples (default: 0.45)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--filename_prefix', type=str, default='phi_query_classification',
                       help='Prefix for output files (default: phi_query_classification)')
    
    args = parser.parse_args()
    
    print("🔧 Phi Synthetic Data Generator")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  Total samples: {args.num_samples}")
    print(f"  Local ratio: {args.local_ratio}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Random seed: {args.random_seed}")
    print()
    
    # Initialize generator
    generator = PhiSyntheticDataGenerator(random_seed=args.random_seed)
    
    # Generate dataset
    print("Starting data generation...")
    dataset = generator.generate_dataset(args.num_samples, args.local_ratio)
    
    # Save dataset
    print("\nSaving dataset...")
    file_paths = generator.save_dataset(dataset, args.output_dir, args.filename_prefix)
    
    # Print summary
    print("\n✅ Data generation completed successfully!")
    print(f"Generated {len(dataset)} samples for Phi model fine-tuning")
    print("\nFiles created:")
    for file_type, path in file_paths.items():
        print(f"  {file_type}: {path}")
    
    print(f"\nNext steps:")
    print(f"1. Review the generated data in: {file_paths['analysis_file']}")
    print(f"2. Use the training file for Phi fine-tuning: {file_paths['training_file']}")
    print(f"3. Run the fine-tuning script: python finetune_phi_router.py")


if __name__ == "__main__":
    main()