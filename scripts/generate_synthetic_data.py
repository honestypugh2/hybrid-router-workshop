"""
Synthetic Data Generator for BERT-based Query Router

This script generates synthetic training data for classifying user queries
as either 'local' (simple queries) or 'cloud' (complex queries).
"""

import json
import random
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple


class SyntheticDataGenerator:
    """Generates synthetic queries for training a BERT-based router."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with a random seed for reproducibility."""
        random.seed(seed)
        
        # Local query templates (simple, fast responses)
        self.local_templates = {
            'greetings': [
                "Hello", "Hi", "Hey", "Good morning", "Good afternoon", 
                "Good evening", "How are you", "What's up", "Greetings"
            ],
            'simple_math': [
                "What is {num1} + {num2}?", "Calculate {num1} * {num2}",
                "What's {num1} - {num2}?", "Divide {num1} by {num2}",
                "What is {num1} to the power of {num2}?", "Find {num1}% of {num2}"
            ],
            'simple_facts': [
                "What is the capital of {country}?", "Who invented the {invention}?",
                "When was {item} created?", "What does {acronym} stand for?",
                "What is the population of {city}?", "How tall is {landmark}?"
            ],
            'definitions': [
                "Define {term}", "What is {concept}?", "Meaning of {word}",
                "What does {technical_term} mean?", "Explain {simple_concept}"
            ],
            'conversions': [
                "Convert {value} {unit1} to {unit2}", "How many {unit1} in {value} {unit2}?",
                "Change {temp}°F to Celsius", "Convert {distance} miles to kilometers"
            ],
            'yes_no_questions': [
                "Is {item} {adjective}?", "Can {subject} {verb}?",
                "Does {entity} have {feature}?", "Is it true that {statement}?"
            ]
        }
        
        # Cloud query templates (complex, requiring analysis)
        self.cloud_templates = {
            'analysis': [
                "Analyze the impact of {topic} on {domain}",
                "Compare and contrast {item1} and {item2}",
                "Evaluate the pros and cons of {subject}",
                "Assess the effectiveness of {strategy} in {context}",
                "Examine the relationship between {factor1} and {factor2}"
            ],
            'comprehensive_explanations': [
                "Explain in detail how {process} works",
                "Provide a comprehensive overview of {topic}",
                "Describe the methodology for {approach}",
                "Give a thorough explanation of {concept}",
                "Detail the step-by-step process of {procedure}"
            ],
            'creative_tasks': [
                "Write a {length} {type} about {topic}",
                "Create a {format} for {purpose}",
                "Compose a {style} piece on {subject}",
                "Draft a {document_type} addressing {issue}",
                "Generate creative ideas for {challenge}"
            ],
            'strategic_planning': [
                "Develop a strategy for {objective}",
                "Create a business plan for {venture}",
                "Design a framework for {goal}",
                "Formulate an approach to {problem}",
                "Outline a roadmap for {project}"
            ],
            'research_tasks': [
                "Research the current trends in {field}",
                "Investigate the causes of {phenomenon}",
                "Study the effects of {intervention} on {outcome}",
                "Explore the potential applications of {technology}",
                "Examine the historical development of {subject}"
            ],
            'complex_reasoning': [
                "Why does {phenomenon} occur in {context}?",
                "How might {change} affect {system} in the long term?",
                "What are the implications of {event} for {stakeholder}?",
                "Under what conditions would {scenario} be optimal?",
                "What factors contribute to {outcome} in {situation}?"
            ]
        }
        
        # Entity lists for template filling
        self.entities = {
            'countries': ['France', 'Japan', 'Brazil', 'Canada', 'Australia', 'Germany', 'India', 'China'],
            'cities': ['Paris', 'Tokyo', 'New York', 'London', 'Sydney', 'Berlin', 'Mumbai', 'Beijing'],
            'inventions': ['telephone', 'computer', 'internet', 'airplane', 'light bulb', 'radio'],
            'landmarks': ['Eiffel Tower', 'Mount Everest', 'Statue of Liberty', 'Great Wall of China'],
            'technologies': ['artificial intelligence', 'blockchain', 'quantum computing', 'machine learning'],
            'fields': ['healthcare', 'education', 'finance', 'technology', 'environment', 'transportation'],
            'concepts': ['democracy', 'capitalism', 'sustainability', 'innovation', 'globalization'],
            'units': ['meters', 'feet', 'kilograms', 'pounds', 'liters', 'gallons'],
            'numbers': list(range(1, 101)),
            'temperatures': list(range(-20, 121)),
            'distances': list(range(1, 1001))
        }

    def fill_template(self, template: str) -> str:
        """Fill a template with random entities."""
        filled = template
        
        # Replace placeholders with random entities
        replacements = {
            '{country}': random.choice(self.entities['countries']),
            '{city}': random.choice(self.entities['cities']),
            '{invention}': random.choice(self.entities['inventions']),
            '{landmark}': random.choice(self.entities['landmarks']),
            '{technology}': random.choice(self.entities['technologies']),
            '{field}': random.choice(self.entities['fields']),
            '{concept}': random.choice(self.entities['concepts']),
            '{topic}': random.choice(self.entities['technologies'] + self.entities['concepts']),
            '{domain}': random.choice(self.entities['fields']),
            '{subject}': random.choice(self.entities['concepts'] + self.entities['technologies']),
            '{num1}': str(random.choice(self.entities['numbers'])),
            '{num2}': str(random.choice(self.entities['numbers'])),
            '{value}': str(random.choice(self.entities['numbers'])),
            '{temp}': str(random.choice(self.entities['temperatures'])),
            '{distance}': str(random.choice(self.entities['distances'])),
            '{unit1}': random.choice(self.entities['units']),
            '{unit2}': random.choice(self.entities['units']),
            '{length}': random.choice(['short', 'brief', 'detailed', 'comprehensive']),
            '{type}': random.choice(['story', 'poem', 'essay', 'report', 'article']),
            '{format}': random.choice(['plan', 'framework', 'strategy', 'proposal', 'design']),
            '{purpose}': random.choice(['marketing', 'training', 'research', 'analysis', 'presentation']),
            '{style}': random.choice(['professional', 'creative', 'technical', 'academic', 'casual']),
            '{document_type}': random.choice(['proposal', 'report', 'strategy', 'analysis', 'plan']),
            '{objective}': random.choice(['growth', 'efficiency', 'innovation', 'sustainability', 'expansion']),
            '{venture}': random.choice(['startup', 'product launch', 'service expansion', 'market entry']),
            '{goal}': random.choice(['improvement', 'optimization', 'transformation', 'development']),
            '{problem}': random.choice(['inefficiency', 'complexity', 'scalability', 'accessibility']),
            '{project}': random.choice(['implementation', 'migration', 'upgrade', 'rollout']),
            '{phenomenon}': random.choice(['inflation', 'climate change', 'urbanization', 'digitization']),
            '{intervention}': random.choice(['policy change', 'new technology', 'training program']),
            '{outcome}': random.choice(['productivity', 'satisfaction', 'performance', 'quality']),
            '{stakeholder}': random.choice(['businesses', 'consumers', 'governments', 'employees']),
            '{event}': random.choice(['economic crisis', 'technological breakthrough', 'policy reform']),
            '{system}': random.choice(['economy', 'environment', 'healthcare system', 'education']),
            '{scenario}': random.choice(['remote work', 'automation', 'renewable energy']),
            '{situation}': random.choice(['competitive markets', 'economic uncertainty', 'rapid growth']),
            '{context}': random.choice(['developing countries', 'urban areas', 'digital environments']),
            '{change}': random.choice(['demographic shift', 'technological advancement', 'policy update']),
            '{challenge}': random.choice(['cost reduction', 'user engagement', 'market penetration']),
            '{approach}': random.choice(['agile methodology', 'data-driven strategy', 'user-centric design']),
            '{procedure}': random.choice(['quality assurance', 'risk assessment', 'performance evaluation']),
            '{process}': random.choice(['machine learning', 'photosynthesis', 'supply chain management']),
            '{strategy}': random.choice(['digital transformation', 'market penetration', 'cost leadership']),
            '{factor1}': random.choice(['education level', 'income', 'age', 'location']),
            '{factor2}': random.choice(['job satisfaction', 'health outcomes', 'productivity', 'happiness']),
            '{item}': random.choice(['smartphone', 'computer', 'car', 'book', 'building']),
            '{item1}': random.choice(['iOS', 'democracy', 'solar energy', 'remote work']),
            '{item2}': random.choice(['Android', 'autocracy', 'fossil fuels', 'office work']),
            '{adjective}': random.choice(['efficient', 'reliable', 'sustainable', 'innovative']),
            '{verb}': random.choice(['adapt', 'evolve', 'improve', 'scale', 'transform']),
            '{entity}': random.choice(['company', 'government', 'organization', 'system']),
            '{feature}': random.choice(['security measures', 'user interface', 'scalability', 'flexibility']),
            '{statement}': random.choice(['AI will replace human jobs', 'renewable energy is cost-effective']),
            '{term}': random.choice(['algorithm', 'blockchain', 'API', 'database', 'framework']),
            '{word}': random.choice(['sustainability', 'optimization', 'innovation', 'efficiency']),
            '{technical_term}': random.choice(['machine learning', 'cloud computing', 'data analytics']),
            '{simple_concept}': random.choice(['photosynthesis', 'gravity', 'democracy', 'inflation']),
            '{acronym}': random.choice(['API', 'CPU', 'GPS', 'HTML', 'HTTP', 'JSON', 'SQL', 'URL'])
        }
        
        for placeholder, replacement in replacements.items():
            filled = filled.replace(placeholder, str(replacement))
        
        return filled

    def generate_local_queries(self, count: int) -> List[str]:
        """Generate queries that should be routed to local model."""
        queries = []
        
        for _ in range(count):
            # Choose random category and template
            category = random.choice(list(self.local_templates.keys()))
            template = random.choice(self.local_templates[category])
            
            # Fill template with entities
            query = self.fill_template(template)
            queries.append(query)
        
        return queries

    def generate_cloud_queries(self, count: int) -> List[str]:
        """Generate queries that should be routed to cloud model."""
        queries = []
        
        for _ in range(count):
            # Choose random category and template
            category = random.choice(list(self.cloud_templates.keys()))
            template = random.choice(self.cloud_templates[category])
            
            # Fill template with entities
            query = self.fill_template(template)
            queries.append(query)
        
        return queries

    def generate_dataset(self, 
                        local_count: int = 1000, 
                        cloud_count: int = 1000) -> List[Dict[str, str]]:
        """Generate a balanced dataset of local and cloud queries."""
        
        print(f"Generating {local_count} local queries...")
        local_queries = self.generate_local_queries(local_count)
        
        print(f"Generating {cloud_count} cloud queries...")
        cloud_queries = self.generate_cloud_queries(cloud_count)
        
        # Create dataset
        dataset = []
        
        # Add local queries
        for query in local_queries:
            dataset.append({
                'text': query,
                'label': 'local',
                'label_id': 0
            })
        
        # Add cloud queries
        for query in cloud_queries:
            dataset.append({
                'text': query,
                'label': 'cloud',
                'label_id': 1
            })
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        return dataset

    def save_dataset(self, dataset: List[Dict[str, str]], 
                    filename: str = "query_routing_dataset.json"):
        """Save dataset to JSON file."""
        
        # Add metadata
        dataset_with_metadata = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_samples': len(dataset),
                'local_samples': len([d for d in dataset if d['label'] == 'local']),
                'cloud_samples': len([d for d in dataset if d['label'] == 'cloud']),
                'description': 'Synthetic dataset for training BERT-based query router'
            },
            'data': dataset
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset_with_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset saved to {filename}")
        return filename

    def create_train_test_split(self, dataset: List[Dict[str, str]], 
                               train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into train and test sets."""
        
        # Separate by label to ensure balanced split
        local_data = [d for d in dataset if d['label'] == 'local']
        cloud_data = [d for d in dataset if d['label'] == 'cloud']
        
        # Split each label separately
        local_train_size = int(len(local_data) * train_ratio)
        cloud_train_size = int(len(cloud_data) * train_ratio)
        
        train_data = (local_data[:local_train_size] + 
                     cloud_data[:cloud_train_size])
        test_data = (local_data[local_train_size:] + 
                    cloud_data[cloud_train_size:])
        
        # Shuffle both sets
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        return train_data, test_data


def main():
    """Generate synthetic dataset for BERT-based query router."""
    print("🤖 Synthetic Data Generator for BERT-based Query Router")
    print("=" * 60)
    
    # Initialize generator
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate dataset
    print("\n📊 Generating synthetic dataset...")
    dataset = generator.generate_dataset(local_count=2000, cloud_count=2000)
    
    # Split into train/test
    print("🔄 Creating train/test split...")
    train_data, test_data = generator.create_train_test_split(dataset, train_ratio=0.8)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")
    
    local_train = len([d for d in train_data if d['label'] == 'local'])
    cloud_train = len([d for d in train_data if d['label'] == 'cloud'])
    local_test = len([d for d in test_data if d['label'] == 'local'])
    cloud_test = len([d for d in test_data if d['label'] == 'cloud'])
    
    print(f"   Train - Local: {local_train}, Cloud: {cloud_train}")
    print(f"   Test - Local: {local_test}, Cloud: {cloud_test}")
    
    # Save datasets
    print("\n💾 Saving datasets...")
    
    # Save full dataset
    generator.save_dataset(dataset, "query_routing_full_dataset.json")
    
    # Save train set
    train_dataset = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_samples': len(train_data),
            'local_samples': local_train,
            'cloud_samples': cloud_train,
            'description': 'Training set for BERT-based query router',
            'split': 'train'
        },
        'data': train_data
    }
    
    with open("query_routing_train.json", 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2, ensure_ascii=False)
    
    # Save test set
    test_dataset = {
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_samples': len(test_data),
            'local_samples': local_test,
            'cloud_samples': cloud_test,
            'description': 'Test set for BERT-based query router',
            'split': 'test'
        },
        'data': test_data
    }
    
    with open("query_routing_test.json", 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f, indent=2, ensure_ascii=False)
    
    print("✅ Training dataset saved to: query_routing_train.json")
    print("✅ Test dataset saved to: query_routing_test.json")
    print("✅ Full dataset saved to: query_routing_full_dataset.json")
    
    # Show some examples
    print("\n📝 Sample Queries:")
    print("\nLocal Examples:")
    for i, sample in enumerate(train_data[:3]):
        if sample['label'] == 'local':
            print(f"   {i+1}. {sample['text']}")
    
    print("\nCloud Examples:")
    cloud_count = 0
    for sample in train_data:
        if sample['label'] == 'cloud' and cloud_count < 3:
            print(f"   {cloud_count+1}. {sample['text']}")
            cloud_count += 1
    
    print(f"\n🎉 Synthetic dataset generation complete!")
    print(f"Ready for BERT model training.")


if __name__ == "__main__":
    main()