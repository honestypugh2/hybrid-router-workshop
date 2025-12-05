"""
Hybrid Foundry APIM Router Module

This module provides a comprehensive three-tier hybrid routing system that intelligently
routes queries between Local models (Foundry Local), Azure API Management (APIM) Model Router,
and Azure AI Foundry Agents based on query complexity and enterprise requirements.

NEW: Now uses the new Foundry Agent Service API (responses.create with conversations)
- Modern API with better performance and reliability
- Simplified agent interaction pattern
- Enhanced conversation management

ConversationContextManager integration:
- Session-based conversation management
- Multi-turn context preservation across routing decisions
- Enhanced metadata and analytics
- Conversation history export/import capabilities

Classes:
    HybridRouterConfig: Configuration for the hybrid router system
    HybridFoundryAPIMRouter: Main router class with ML-powered analysis and context management
    
Functions:
    analyze_query_for_hybrid_routing: ML-enhanced query analysis
    route_hybrid_query: Intelligent routing decision logic
    route_with_context: Enhanced routing with conversation context
    create_hybrid_router_from_env: Convenience function to create router from environment variables

Usage Example:
    from modules.hybrid_router import create_hybrid_router_from_env
    
    # Create router with conversation context
    router = create_hybrid_router_from_env(session_id="my_session")
    
    # Route query with context
    result = router.route_with_context("Hello, how are you?", use_context=True)
    print(result['response'])
    
    # Get conversation history
    history = router.get_conversation_history(count=3)
    
    # Clear context when needed
    router.clear_conversation_context()
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

# Import ConversationContextManager
from .context_manager import ConversationContextManager

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

# Optional Azure AI Foundry Agents (new Foundry Agent Service API)
try:
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import PromptAgentDefinition
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
    
    def __init__(self, config: HybridRouterConfig, session_id: str = None):
        """Initialize the hybrid router with configuration."""
        self.config = config
        self.routing_history = []
        
        # Initialize conversation context manager
        self.context_manager = ConversationContextManager(
            session_id=session_id, 
            max_history=config.max_context_length if hasattr(config, 'max_context_length') else 15
        )
        
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
        
        # Azure AI Foundry client with endpoint validation
        if FOUNDRY_AGENTS_AVAILABLE and self.config.foundry_endpoint:
            try:
                # Validate and fix endpoint URL
                foundry_endpoint = self.config.foundry_endpoint
                
                # Clean up endpoint (remove comments, quotes, fix double https)
                foundry_endpoint = foundry_endpoint.split('#')[0].strip()
                foundry_endpoint = foundry_endpoint.strip('"\'')
                
                # Fix double https issue
                if foundry_endpoint.count('https://') > 1:
                    second_https = foundry_endpoint.find('https://', foundry_endpoint.find('https://') + 1)
                    if second_https != -1:
                        foundry_endpoint = foundry_endpoint[second_https:]
                
                # Ensure HTTPS for bearer token authentication
                if foundry_endpoint.startswith('http://'):
                    foundry_endpoint = foundry_endpoint.replace('http://', 'https://')
                elif not foundry_endpoint.startswith('https://'):
                    foundry_endpoint = f"https://{foundry_endpoint}"
                
                # Update config with cleaned endpoint
                self.config.foundry_endpoint = foundry_endpoint
                
                credential = DefaultAzureCredential()
                self.project_client = AIProjectClient(
                    endpoint=foundry_endpoint,
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
        """Initialize Foundry Agent using new Foundry Agent Service API."""
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
                
                # Use new Foundry Agent Service API
                self.foundry_agent = self.project_client.agents.create_version(
                    agent_name="Hybrid-Router-Agent",
                    definition=PromptAgentDefinition(
                        model=self.config.azure_deployment,
                        instructions=agent_instructions.strip()
                    )
                )
                
                # Create conversation thread using new API
                ai_openai_client = self.project_client.get_openai_client()
                self.agent_thread = ai_openai_client.conversations.create()
                print("✅ Foundry Agent created (new API)")
                
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
        """Query Azure AI Foundry Agent using new Foundry Agent Service API."""
        if not self.foundry_agent or not self.agent_thread:
            return "Foundry Agent not available", 0, False
        
        try:
            start_time = time.time()
            
            # Get the OpenAI client from project_client for conversations API
            ai_openai_client = self.project_client.get_openai_client()
            
            # Chat with the agent using responses.create (new Foundry Agent Service API)
            response = ai_openai_client.responses.create(
                conversation=self.agent_thread.id,
                extra_body={
                    "agent": {
                        "name": self.foundry_agent.name,
                        "type": "agent_reference"
                    }
                },
                input=prompt
            )
            
            end_time = time.time()
            
            # Extract the response text
            if hasattr(response, 'output_text'):
                content = response.output_text
            else:
                content = str(response)
            
            return content, end_time - start_time, True
                
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
    
    def route_with_context(self, query: str, use_context: bool = True, show_reasoning: bool = False) -> Dict[str, Any]:
        """
        Process query through hybrid routing system with conversation context.
        
        Args:
            query: User query to route
            use_context: Whether to include conversation context
            show_reasoning: Whether to include routing reasoning in response
            
        Returns:
            Dictionary with response, metadata, routing info, and context details
        """
        start_time = time.time()
        
        # Get conversation context if requested
        context_messages = []
        if use_context:
            context_messages = self.context_manager.get_messages_for_model('both', include_system=False)
        
        # Prepare query with context for model input
        query_with_context = query
        if context_messages:
            # Format recent context for the model
            context_str = self.context_manager.get_conversation_context(context_messages)
            if context_str != "No previous context available.":
                query_with_context = f"{context_str}\n\nCurrent question: {query}"
        
        # Determine optimal routing target
        target, reason, priority = self.route_hybrid_query(query)
        
        # Get routing analysis for metadata
        analysis = self.analyze_query_for_hybrid_routing(query)
        
        response = ""
        response_time = 0
        success = False
        actual_source = target
        
        # Execute primary routing decision with fallback chains
        if target == 'local':
            response, response_time, success = self.query_local_model(query_with_context)
            if not success:
                if self.apim_client:
                    response, response_time, success = self.query_apim_router(query_with_context)
                    actual_source = 'apim-fallback'
                elif self.foundry_agent:
                    response, response_time, success = self.query_foundry_agent(query_with_context)
                    actual_source = 'foundry-fallback'
                elif self.azure_client:
                    response, response_time, success = self.query_azure_direct(query_with_context)
                    actual_source = 'azure-fallback'
        
        elif target == 'apim':
            response, response_time, success = self.query_apim_router(query_with_context)
            if not success:
                if self.foundry_agent:
                    response, response_time, success = self.query_foundry_agent(query_with_context)
                    actual_source = 'foundry-fallback'
                elif self.azure_client:
                    response, response_time, success = self.query_azure_direct(query_with_context)
                    actual_source = 'azure-fallback'
                elif self.local_client:
                    response, response_time, success = self.query_local_model(query_with_context)
                    actual_source = 'local-fallback'
        
        elif target == 'foundry':
            response, response_time, success = self.query_foundry_agent(query_with_context)
            if not success:
                if self.apim_client:
                    response, response_time, success = self.query_apim_router(query_with_context)
                    actual_source = 'apim-fallback'
                elif self.azure_client:
                    response, response_time, success = self.query_azure_direct(query_with_context)
                    actual_source = 'azure-fallback'
                elif self.local_client:
                    response, response_time, success = self.query_local_model(query_with_context)
                    actual_source = 'local-fallback'
        
        total_time = time.time() - start_time
        
        # Clean response content (remove source tags for storage)
        clean_response = response
        source_tags = ['[LOCAL]', '[APIM]', '[FOUNDRY-AGENT]', '[APIM*]', '[FOUNDRY*]', '[AZURE*]', '[LOCAL*]', '[ERROR]']
        for tag in source_tags:
            if clean_response.startswith(tag):
                clean_response = clean_response[len(tag):].strip()
                break
        
        # Create metadata for the exchange
        metadata = {
            'strategy': target,
            'context_used': use_context and len(context_messages) > 0,
            'routing_info': {
                'target': target,
                'actual_source': actual_source,
                'reason': reason,
                'priority': priority,
                'strategy': target,
                'analysis': analysis,
                'confidence': analysis.get('ml_confidence', 0.0),
                'router_used': analysis.get('router_used', 'pattern_based'),
                'source': actual_source
            },
            'response_time': total_time,
            'success': success,
            'context_length': len(context_messages),
            'query_length': len(query),
            'routing_decision': reason,
            'router_used': analysis.get('router_used', 'pattern_based')
        }
        
        if not success:
            metadata['error'] = True
            metadata['error_message'] = f"Routing failed: {response}"
        
        # Add exchange to conversation context
        self.context_manager.add_exchange(
            user_message=query,
            ai_response=clean_response,
            source=actual_source,
            response_time=total_time,
            metadata=metadata
        )
        
        # Record for statistics
        self.routing_history.append({
            'query': query,
            'source': actual_source,
            'response_time': response_time,
            'success': success,
            'timestamp': time.time(),
            'context_used': use_context,
            'metadata': metadata
        })
        
        # Format final response
        source_tags_dict = {
            'local': '[LOCAL]',
            'apim': '[APIM]',
            'foundry': '[FOUNDRY-AGENT]',
            'apim-fallback': '[APIM*]',
            'foundry-fallback': '[FOUNDRY*]',
            'azure-fallback': '[AZURE*]',
            'local-fallback': '[LOCAL*]'
        }
        
        if success:
            source_tag = source_tags_dict.get(actual_source, f'[{actual_source.upper()}]')
            formatted_response = f"{source_tag} {clean_response}"
            if show_reasoning:
                formatted_response += f"\n\n[Routing: {reason}]"
        else:
            formatted_response = f"[ERROR] All routing options failed: {response}"
        
        return {
            'response': clean_response,
            'formatted_response': formatted_response,
            'source': actual_source,
            'responseTime': total_time,
            'metadata': metadata,
            'success': success
        }

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
    
    # Conversation Context Management Methods
    def get_conversation_history(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation exchanges."""
        return self.context_manager.get_recent_exchanges(count)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get comprehensive session summary including routing and conversation stats."""
        context_summary = self.context_manager.get_session_summary()
        routing_stats = self.get_comprehensive_stats()
        
        return {
            'session_info': context_summary.get('session_info', {}),
            'routing_stats': context_summary.get('routing_stats', {}),
            'conversation_flow': context_summary.get('conversation_flow', []),
            'hybrid_routing_stats': routing_stats,
            'context_manager_stats': self.context_manager.get_conversation_summary()
        }
    
    def clear_conversation_context(self):
        """Clear conversation history and reset context."""
        self.context_manager.clear_conversation()
        print(f"🧹 Conversation context cleared for session {self.context_manager.session_id}")
    
    def export_conversation_history(self, filename: str = None) -> str:
        """Export conversation history to JSON file."""
        return self.context_manager.export_conversation(filename)
    
    def get_context_for_query(self, max_messages: int = 5) -> str:
        """Get formatted conversation context for query processing."""
        recent_messages = self.context_manager.get_messages_for_model('both', include_system=False)
        return self.context_manager.get_conversation_context(recent_messages[-max_messages:])


# Convenience functions for backwards compatibility
def create_hybrid_router_from_env(session_id: str = None) -> HybridFoundryAPIMRouter:
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
    
    return HybridFoundryAPIMRouter(config, session_id=session_id)


# Test function for ConversationContextManager integration
def test_context_integration():
    """Test the integration between HybridFoundryAPIMRouter and ConversationContextManager."""
    print("🧪 Testing ConversationContextManager integration...")
    
    # Create a minimal config for testing
    config = HybridRouterConfig(
        local_endpoint="http://localhost:1234",
        local_model_id="test-model"
    )
    
    # Create router with context management
    router = HybridFoundryAPIMRouter(config, session_id="test_session")
    
    print(f"✅ Router created with session: {router.context_manager.session_id}")
    print(f"✅ Context manager initialized: {type(router.context_manager).__name__}")
    
    # Test conversation history methods
    history = router.get_conversation_history()
    print(f"✅ Initial conversation history: {len(history)} exchanges")
    
    # Test session summary
    summary = router.get_session_summary()
    print(f"✅ Session summary keys: {list(summary.keys())}")
    
    # Test context retrieval
    context = router.get_context_for_query()
    print(f"✅ Context for query: {context}")
    
    print("🎉 ConversationContextManager integration test completed!")


if __name__ == "__main__":
    test_context_integration()