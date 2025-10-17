"""
Hybrid Foundry APIM Router Module

This module provides a comprehensive three-tier hybrid routing system that intelligently
routes queries between Local models (Foundry Local), Azure API Management (APIM) Model Router,
and Azure AI Foundry Agents based on query complexity and enterprise requirements.

Classes:
    HybridRouterConfig: Configuration for the hybrid router system
    HybridFoundryAPIMRouter: Main router class with ML-powered analysis
    
Functions:
    analyze_query_for_hybrid_routing: ML-enhanced query analysis
    route_hybrid_query: Intelligent routing decision logic
"""

import os
import time
import re
import requests
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# OpenAI clients
from openai import OpenAI, AzureOpenAI

# Optional ML router imports
try:
    from .bert_router import BertQueryRouter, BertRouterConfig
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False

try:
    from .phi_router import PhiQueryRouter, PhiRouterConfig
    PHI_AVAILABLE = True
except ImportError:
    PHI_AVAILABLE = False

# Optional Azure AI Foundry Agents
try:
    from azure.ai.projects import AIProjectClient
    from azure.ai.agents.models import MessageRole, RunStatus
    from azure.identity import DefaultAzureCredential
    FOUNDRY_AGENTS_AVAILABLE = True
except ImportError:
    FOUNDRY_AGENTS_AVAILABLE = False


@dataclass
class HybridRouterConfig:
    """Configuration for the hybrid router system."""
    # Local model configuration
    local_endpoint: Optional[str] = None
    local_model_name: Optional[str] = None
    local_model_id: Optional[str] = None
    
    # APIM configuration
    apim_endpoint: Optional[str] = None
    apim_key: Optional[str] = None
    apim_deployment_id: Optional[str] = None
    
    # Azure AI Foundry configuration
    foundry_endpoint: Optional[str] = None
    
    # Azure OpenAI fallback configuration
    azure_endpoint: Optional[str] = None
    azure_key: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_api_version: str = "2024-12-01-preview"
    
    # ML router configuration
    bert_model_path: Optional[str] = os.environ["BERT_MODEL_FULLPATH"]
    phi_model_path: Optional[str] = os.environ["PHI_MODEL_FULLPATH"]
    ml_confidence_threshold: float = float(os.environ["ML_CONFIDENCE_THRESHOLD"])
    
    # Routing thresholds
    complexity_threshold_apim: int = 5
    complexity_threshold_foundry: int = 3
    enterprise_word_threshold: int = 30


class HybridFoundryAPIMRouter:
    """
    Advanced hybrid router: Local (Foundry Local) → APIM → Foundry Agents → Azure.
    
    Provides intelligent three-tier routing with ML-powered query analysis,
    enterprise-grade APIM integration, and robust fallback mechanisms.
    """
    
    def __init__(self, config: HybridRouterConfig):
        """Initialize the hybrid router with configuration."""
        self.config = config
        self.routing_history = []
        
        # Initialize ML routers
        self.bert_router = None
        self.phi_router = None
        self._init_ml_routers()
        
        # Initialize clients
        self.local_client = None
        self.apim_client = None
        self.project_client = None
        self.azure_client = None
        self.foundry_agent = None
        self.agent_thread = None
        
        self._init_clients()
        self._init_foundry_agent()
    
    def _init_ml_routers(self):
        """Initialize ML-based routers if available."""
        # Initialize BERT router
        if BERT_AVAILABLE and self.config.bert_model_path:
            try:
                bert_config = BertRouterConfig(
                    model_path=self.config.bert_model_path,
                    confidence_threshold=self.config.ml_confidence_threshold
                )
                self.bert_router = BertQueryRouter(bert_config)
                self.bert_router._load_model()
                print("✅ BERT router initialized")
            except Exception as e:
                print(f"⚠️ BERT router initialization failed: {e}")
        
        # Initialize PHI router
        if PHI_AVAILABLE and self.config.phi_model_path and self.bert_router is None:
            try:
                phi_config = PhiRouterConfig(
                    model_path=self.config.phi_model_path,
                    confidence_threshold=self.config.ml_confidence_threshold
                )
                self.phi_router = PhiQueryRouter(phi_config)
                self.phi_router._load_model()
                print("✅ PHI router initialized")
            except Exception as e:
                print(f"⚠️ PHI router initialization failed: {e}")
    
    def _init_clients(self):
        """Initialize API clients for different routing targets."""
        # Local client
        if self.config.local_endpoint:
            try:
                self.local_client = OpenAI(
                    base_url=f"{self.config.local_endpoint}/v1",
                    api_key="not-needed"
                )
                print("✅ Local client initialized")
            except Exception as e:
                print(f"⚠️ Local client failed: {e}")
        
        # APIM client
        if self.config.apim_endpoint and self.config.apim_key:
            try:
                self.apim_client = AzureOpenAI(
                    azure_endpoint=self.config.apim_endpoint,
                    api_key=self.config.apim_key,
                    api_version=self.config.azure_api_version
                )
                print("✅ APIM client initialized")
            except Exception as e:
                print(f"⚠️ APIM client failed: {e}")
        
        # Azure AI Foundry client
        if FOUNDRY_AGENTS_AVAILABLE and self.config.foundry_endpoint:
            try:
                credential = DefaultAzureCredential()
                self.project_client = AIProjectClient(
                    endpoint=self.config.foundry_endpoint,
                    credential=credential
                )
                print("✅ Azure AI Foundry client initialized")
            except Exception as e:
                print(f"⚠️ Foundry client failed: {e}")
        
        # Azure OpenAI fallback
        if self.config.azure_endpoint and self.config.azure_key:
            try:
                self.azure_client = AzureOpenAI(
                    api_key=self.config.azure_key,
                    api_version=self.config.azure_api_version,
                    azure_endpoint=self.config.azure_endpoint
                )
                print("✅ Azure OpenAI client initialized")
            except Exception as e:
                print(f"⚠️ Azure OpenAI client failed: {e}")
    
    def _init_foundry_agent(self):
        """Initialize Foundry Agent if available."""
        if self.project_client and self.config.azure_deployment:
            try:
                agent_instructions = """
                You are an intelligent AI assistant in a hybrid local-cloud system.
                You handle complex queries requiring:
                - Advanced reasoning and analysis
                - Multi-turn conversation management
                - Creative content generation
                - Strategic planning and recommendations
                - Document analysis and summarization
                
                Provide clear, comprehensive responses while being efficient.
                """
                
                self.foundry_agent = self.project_client.agents.create_agent(
                    model=self.config.azure_deployment,
                    name="Hybrid-Router-Agent",
                    instructions=agent_instructions.strip(),
                    description="Specialized agent for complex queries in hybrid AI system"
                )
                
                self.agent_thread = self.project_client.agents.threads.create()
                print("✅ Foundry Agent created")
                
            except Exception as e:
                print(f"⚠️ Failed to create Foundry Agent: {e}")
    
    def analyze_query_for_hybrid_routing(self, query: str) -> Dict:
        """Enhanced query analysis using BERT/PHI routers for three-tier hybrid routing."""
        query_lower = query.lower().strip()
        
        analysis = {
            'original_query': query,
            'length': len(query),
            'word_count': len(query.split()),
            'is_greeting': False,
            'is_simple_question': False,
            'is_calculation': False,
            'requires_analysis': False,
            'requires_creativity': False,
            'is_conversational': False,
            'is_enterprise_query': False,
            'complexity_score': 0,
            'router_used': 'bert',
            'reasoning': '',
            'ml_prediction': None,
            'ml_confidence': 0.0
        }
        
        # Use BERT router if available (preferred)
        if self.bert_router is not None:
            try:
                target, reason, metadata = self.bert_router.route_query(query)
                analysis['router_used'] = 'bert'
                analysis['reasoning'] = reason
                analysis['ml_prediction'] = target
                analysis['ml_confidence'] = metadata.get('confidence', 0.0)
                
                # Map BERT predictions to hybrid routing decisions
                if target == 'local':
                    analysis['complexity_score'] = max(0, 3 - int(metadata.get('confidence', 0) * 5))
                else:  # cloud
                    analysis['complexity_score'] = min(10, 5 + int(metadata.get('confidence', 0) * 5))
                
            except Exception as e:
                print(f"⚠️ BERT router failed, falling back: {e}")
                analysis['router_used'] = 'pattern_based_fallback'
        
        # Use PHI router if BERT not available
        elif self.phi_router is not None:
            try:
                target, reason, metadata = self.phi_router.route_query(query)
                analysis['router_used'] = 'phi'
                analysis['reasoning'] = reason
                analysis['ml_prediction'] = target
                analysis['ml_confidence'] = metadata.get('confidence', 0.0)
                
                # Map PHI predictions to hybrid routing decisions
                if target == 'local':
                    analysis['complexity_score'] = max(0, 4 - int(metadata.get('confidence', 0) * 6))
                else:  # cloud
                    analysis['complexity_score'] = min(10, 4 + int(metadata.get('confidence', 0) * 6))
                
            except Exception as e:
                print(f"⚠️ PHI router failed, falling back: {e}")
                analysis['router_used'] = 'pattern_based_fallback'
        
        # Fallback to pattern-based analysis
        if analysis['router_used'].endswith('fallback') or (self.bert_router is None and self.phi_router is None):
            self._pattern_based_analysis(query_lower, analysis)
        
        return analysis
    
    def _pattern_based_analysis(self, query_lower: str, analysis: Dict):
        """Perform pattern-based query analysis."""
        # Pattern detection
        greeting_patterns = [r'^(hi|hello|hey|good morning|good afternoon|good evening)']
        simple_patterns = [r'^(what is|who is|where is|when is|how much|what time)']
        calc_patterns = [r'\d+\s*[+\-*/]\s*\d+', r'calculate|compute|solve']
        
        # Complex task indicators
        complex_keywords = ['analyze', 'summarize', 'explain in detail', 'comprehensive',
                           'compare', 'evaluate', 'strategy', 'plan', 'implications']
        creative_keywords = ['write a', 'create a', 'compose', 'design', 'brainstorm', 'imagine']
        enterprise_keywords = ['business', 'enterprise', 'production', 'scalable', 'architecture',
                              'compliance', 'security', 'deployment', 'infrastructure']
        
        # Apply pattern matching
        for pattern in greeting_patterns:
            if re.match(pattern, query_lower):
                analysis['is_greeting'] = True
                break
        
        for pattern in simple_patterns:
            if re.match(pattern, query_lower):
                analysis['is_simple_question'] = True
                break
        
        for pattern in calc_patterns:
            if re.search(pattern, query_lower):
                analysis['is_calculation'] = True
                break
        
        # Check for complex keywords
        analysis['requires_analysis'] = any(kw in query_lower for kw in complex_keywords)
        analysis['requires_creativity'] = any(kw in query_lower for kw in creative_keywords)
        analysis['is_enterprise_query'] = any(kw in query_lower for kw in enterprise_keywords)
        analysis['is_conversational'] = any(word in query_lower for word in 
                                          ['discuss', 'conversation', 'talk about', 'tell me about'])
        
        # Calculate complexity score (pattern-based)
        score = 0
        if analysis['word_count'] > 20: score += 2
        if analysis['requires_analysis']: score += 3
        if analysis['requires_creativity']: score += 2
        if analysis['is_enterprise_query']: score += 2
        if analysis['is_conversational']: score += 1
        if analysis['word_count'] > 50: score += 2
        analysis['complexity_score'] = score
    
    def route_hybrid_query(self, query: str) -> Tuple[str, str, int]:
        """Determine optimal routing target for three-tier hybrid system using ML routers."""
        analysis = self.analyze_query_for_hybrid_routing(query)
        
        # Priority 1: Local (Foundry Local) for simple, fast queries
        if (analysis['ml_prediction'] == 'local' and analysis['ml_confidence'] > 0.8) or \
           (analysis['is_greeting'] or analysis['is_calculation'] or 
            (analysis['is_simple_question'] and analysis['word_count'] <= 8) or
            (analysis['word_count'] <= 5 and not analysis['is_enterprise_query'])):
            return 'local', f"Simple query - local for instant response (via {analysis['router_used']})", 1
        
        # Priority 2: APIM Model Router for enterprise and complex routing
        if self.apim_client and ((analysis['is_enterprise_query']) or 
                              (analysis['complexity_score'] >= self.config.complexity_threshold_apim) or 
                              (analysis['word_count'] > self.config.enterprise_word_threshold) or
                              (analysis['ml_prediction'] == 'cloud' and analysis['ml_confidence'] > 0.8)):
            return 'apim', f"Enterprise/complex query - APIM Model Router (via {analysis['router_used']})", 2
        
        # Priority 3: Foundry Agents for advanced reasoning
        if (analysis['requires_analysis'] or analysis['requires_creativity'] or 
            analysis['is_conversational'] or 
            (analysis['ml_prediction'] == 'cloud' and analysis['ml_confidence'] > 0.6)):
            return 'foundry', f"Complex analysis - Foundry Agent capabilities (via {analysis['router_used']})", 3
        
        # Default routing logic
        if analysis['complexity_score'] <= self.config.complexity_threshold_foundry or analysis['word_count'] <= 12:
            return 'local', f"Default: simple query for local efficiency (via {analysis['router_used']})", 1
        elif self.apim_client:
            return 'apim', f"Default: route to enterprise APIM for quality (via {analysis['router_used']})", 2
        else:
            return 'foundry', f"Default: Foundry Agent for comprehensive response (via {analysis['router_used']})", 3
    
    def query_local_model(self, prompt: str, max_tokens: int = 200) -> Tuple[str, float, bool]:
        """Query local Foundry Local model."""
        if not self.local_client:
            return "Local model not available", 0, False
        
        try:
            start_time = time.time()
            response = self.local_client.chat.completions.create(
                model=self.config.local_model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            end_time = time.time()
            
            content = response.choices[0].message.content
            return content, end_time - start_time, True
        except Exception as e:
            return f"Local model error: {str(e)}", 0, False
    
    def query_apim_router(self, prompt: str) -> Tuple[str, float, bool]:
        """Query through APIM Model Router for enterprise cloud routing."""
        if not self.apim_client:
            return "APIM Model Router not available", 0, False
        
        try:
            start_time = time.time()
            
            # Analyze query for model selection
            analysis = self.analyze_query_for_hybrid_routing(prompt)
            
            # Determine preferred model based on complexity
            preferred_model = "gpt-4.1" if analysis['complexity_score'] >= 6 else "gpt-4o-mini"
            
            # response = self.apim_client.chat.completions.create(
            #     model=self.config.apim_deployment_id,
            #     messages=[{"role": "user", "content": prompt}],
            #     max_tokens=500,
            #     temperature=0.7,
            #     extra_headers={"Preferred-Model": preferred_model}
            # )
            base_url = f"{self.config.apim_endpoint.rstrip('/')}/{self.config.apim_deployment_id}"
    
            headers = {
                'api-key': self.config.apim_key,
                'Content-Type': 'application/json'
            }

            # Test with a standard model name (not deployment ID)
            payload = {
                "model": f"{preferred_model}",  # Use standard model name
                "messages": [
                    {
                        "role": "system", 
                        "content": f"You are an AI assistant. Route this to {preferred_model} model."
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500
            }

            # Use POST method for chat completions
            response = requests.post(base_url, headers=headers, json=payload, timeout=10)
                
            end_time = time.time()
            # content = response.choices[0].message.content
            response_data = response.json()
            content = response_data.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
            return f"[APIM-{preferred_model}] {content}", end_time - start_time, True
            
        except Exception as e:
            return f"APIM Model Router error: {str(e)}", 0, False
    
    def query_foundry_agent(self, prompt: str) -> Tuple[str, float, bool]:
        """Query Azure AI Foundry Agent."""
        if not self.foundry_agent or not self.agent_thread:
            return "Foundry Agent not available", 0, False
        
        try:
            start_time = time.time()
            
            # Add message to thread
            message = self.project_client.agents.messages.create(
                thread_id=self.agent_thread.id,
                role=MessageRole.USER,
                content=prompt
            )
            
            # Create and execute run
            run = self.project_client.agents.runs.create_and_process(
                thread_id=self.agent_thread.id,
                agent_id=self.foundry_agent.id
            )
            
            # Wait for completion
            while run.status in [RunStatus.IN_PROGRESS, RunStatus.QUEUED]:
                time.sleep(0.5)
                run = self.project_client.agents.runs.get(thread_id=self.agent_thread.id, run_id=run.id)
            
            end_time = time.time()
            
            if run.status == RunStatus.COMPLETED:
                messages = self.project_client.agents.messages.list(thread_id=self.agent_thread.id)
   
                # Convert ItemPaged to list and get the most recent message
                message_list = list(messages)
                if message_list:
                    latest_message = message_list[0]  # Most recent message
                    
                    if latest_message.role == MessageRole.ASSISTANT:
                        # Handle different content types
                        if hasattr(latest_message.content[0], 'text'):
                            content = latest_message.content[0].text.value
                        else:
                            content = str(latest_message.content[0])
                        return content, end_time - start_time, True
                    else:
                        return "No assistant response found", end_time - start_time, False
                else:
                    return "No messages found in thread", end_time - start_time, False
            # else:
            #     return f"Agent failed with status: {run.status}", end_time - start_time, False
            
            return f"Agent run failed: {run.status}", end_time - start_time, False
                
        except Exception as e:
            return f"Foundry Agent error: {str(e)}", 0, False
    
    def query_azure_direct(self, prompt: str, max_tokens: int = 400) -> Tuple[str, float, bool]:
        """Query Azure OpenAI directly (final fallback)."""
        if not self.azure_client:
            return "Azure OpenAI not available", 0, False
        
        try:
            start_time = time.time()
            response = self.azure_client.chat.completions.create(
                model=self.config.azure_deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            end_time = time.time()
            
            content = response.choices[0].message.content
            return content, end_time - start_time, True
        except Exception as e:
            return f"Azure OpenAI error: {str(e)}", 0, False
    
    def route(self, query: str, show_reasoning: bool = False) -> str:
        """
        Process query through hybrid three-tier routing system.
        
        Args:
            query: User query to route
            show_reasoning: Whether to include routing reasoning in response
            
        Returns:
            Formatted response with source indication
        """
        # Determine optimal routing target
        target, reason, priority = self.route_hybrid_query(query)
        
        response = ""
        response_time = 0
        success = False
        actual_source = target
        
        # Execute primary routing decision with fallback chains
        if target == 'local':
            response, response_time, success = self.query_local_model(query)
            if not success:
                if self.apim_client:
                    response, response_time, success = self.query_apim_router(query)
                    actual_source = 'apim-fallback'
                elif self.foundry_agent:
                    response, response_time, success = self.query_foundry_agent(query)
                    actual_source = 'foundry-fallback'
                elif self.azure_client:
                    response, response_time, success = self.query_azure_direct(query)
                    actual_source = 'azure-fallback'
        
        elif target == 'apim':
            response, response_time, success = self.query_apim_router(query)
            if not success:
                if self.foundry_agent:
                    response, response_time, success = self.query_foundry_agent(query)
                    actual_source = 'foundry-fallback'
                elif self.azure_client:
                    response, response_time, success = self.query_azure_direct(query)
                    actual_source = 'azure-fallback'
                elif self.local_client:
                    response, response_time, success = self.query_local_model(query)
                    actual_source = 'local-fallback'
        
        elif target == 'foundry':
            response, response_time, success = self.query_foundry_agent(query)
            if not success:
                if self.apim_client:
                    response, response_time, success = self.query_apim_router(query)
                    actual_source = 'apim-fallback'
                elif self.azure_client:
                    response, response_time, success = self.query_azure_direct(query)
                    actual_source = 'azure-fallback'
                elif self.local_client:
                    response, response_time, success = self.query_local_model(query)
                    actual_source = 'local-fallback'
        
        # Format response with source indication
        source_tags = {
            'local': '[LOCAL]',
            'apim': '[APIM]',
            'foundry': '[FOUNDRY-AGENT]',
            'apim-fallback': '[APIM*]',
            'foundry-fallback': '[FOUNDRY*]',
            'azure-fallback': '[AZURE*]',
            'local-fallback': '[LOCAL*]'
        }
        
        if success:
            source_tag = source_tags.get(actual_source, f'[{actual_source.upper()}]')
            formatted_response = f"{source_tag} {response}"
            if show_reasoning:
                formatted_response += f"\n\n[Routing: {reason}]"
        else:
            formatted_response = f"[ERROR] All routing options failed: {response}"
        
        # Record for statistics
        self.routing_history.append({
            'query': query,
            'source': actual_source,
            'response_time': response_time,
            'success': success,
            'timestamp': time.time()
        })
        
        return formatted_response
    
    def get_comprehensive_stats(self) -> Dict:
        """Get detailed routing statistics."""
        if not self.routing_history:
            return {'message': 'No routing history available'}
        
        total = len(self.routing_history)
        successful = sum(1 for h in self.routing_history if h['success'])
        
        # Categorize by primary source
        local_routes = sum(1 for h in self.routing_history if 'local' in h['source'])
        apim_routes = sum(1 for h in self.routing_history if 'apim' in h['source'])
        foundry_routes = sum(1 for h in self.routing_history if 'foundry' in h['source'])
        azure_routes = sum(1 for h in self.routing_history if 'azure' in h['source'])
        
        # Fallback analysis
        fallback_routes = sum(1 for h in self.routing_history if 'fallback' in h['source'])
        
        # Performance metrics
        successful_times = [h['response_time'] for h in self.routing_history if h['success']]
        avg_time = sum(successful_times) / max(len(successful_times), 1)
        
        return {
            'total_queries': total,
            'successful_queries': successful,
            'success_rate': successful / total * 100,
            'routing_distribution': {
                'local': local_routes,
                'apim': apim_routes,
                'foundry': foundry_routes,
                'azure': azure_routes
            },
            'routing_percentages': {
                'local': local_routes / total * 100,
                'apim': apim_routes / total * 100,
                'foundry': foundry_routes / total * 100,
                'azure': azure_routes / total * 100
            },
            'fallback_usage': {
                'count': fallback_routes,
                'percentage': fallback_routes / total * 100
            },
            'performance': {
                'average_response_time': avg_time,
                'fastest_response': min(successful_times) if successful_times else 0,
                'slowest_response': max(successful_times) if successful_times else 0
            }
        }
    
    def get_system_capabilities(self) -> Dict:
        """Get comprehensive system capabilities."""
        return {
            'available_targets': {
                'local_foundry': self.local_client is not None,
                'apim_model_router': self.apim_client is not None,
                'foundry_agents': self.foundry_agent is not None,
                'azure_direct': self.azure_client is not None
            },
            'ml_routers': {
                'bert_router': self.bert_router is not None,
                'phi_router': self.phi_router is not None
            },
            'routing_priorities': [
                'Local (Foundry Local) - Fast, private processing',
                'APIM Model Router - Enterprise routing with load balancing', 
                'Foundry Agents - Advanced reasoning and conversation',
                'Azure Direct - Final fallback option'
            ],
            'fallback_chains': {
                'local_chain': 'Local → APIM → Foundry → Azure',
                'apim_chain': 'APIM → Foundry → Azure → Local',
                'foundry_chain': 'Foundry → APIM → Azure → Local'
            }
        }


# Convenience functions for backwards compatibility
def create_hybrid_router_from_env() -> HybridFoundryAPIMRouter:
    """Create hybrid router using environment variables."""
    config = HybridRouterConfig(
        local_endpoint=os.environ.get("LOCAL_MODEL_ENDPOINT"),
        local_model_name=os.environ.get("LOCAL_MODEL_NAME"),
        local_model_id=os.environ.get("LOCAL_MODEL_ID"),
        apim_endpoint=os.environ.get("APIM_ENDPOINT"),
        apim_key=os.environ.get("APIM_API_KEY"),
        apim_deployment_id=os.environ.get("AZURE_APIM_DEPLOYMENT_ID"),
        foundry_endpoint=os.environ.get("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_key=os.environ.get("AZURE_OPENAI_KEY"),
        azure_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME"),
        azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
    )
    
    return HybridFoundryAPIMRouter(config)