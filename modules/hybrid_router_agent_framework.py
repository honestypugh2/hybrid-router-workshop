"""
Hybrid Router with Agent Framework Module

This module provides a comprehensive two-tier hybrid routing system that intelligently
routes queries between Local models (Foundry Local) and Azure Cloud using Azure API 
Management (APIM) Model Router and Agent Framework with Foundry models based on query 
complexity and enterprise requirements.

NEW: Modern Agent Framework Integration
- Uses agent_framework.azure.AzureAIAgentClient for cloud routing
- Async/await patterns throughout
- Ephemeral and persistent agent support
- Enhanced conversation context management
- ML-powered routing decisions (BERT/PHI)

Classes:
    HybridAgentRouterConfig: Configuration for the hybrid router with Agent Framework
    HybridAgentRouter: Main router class with Agent Framework integration
    
Functions:
    analyze_query_for_routing: ML-enhanced query analysis
    route_query_intelligently: Intelligent routing decision logic
    create_hybrid_agent_router_from_env: Convenience function to create router from environment

Usage Example:
    from modules.hybrid_router_agent_framework import create_hybrid_agent_router_from_env
    import asyncio
    
    # Create router with Agent Framework
    router = create_hybrid_agent_router_from_env(session_id="my_session")
    
    # Route query asynchronously
    async def main():
        result = await router.route_async("Analyze this complex business document...")
        print(result['response'])
    
    asyncio.run(main())
"""

import os
import re
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import asyncio

warnings.filterwarnings('ignore')

# OpenAI clients for local and APIM
from openai import OpenAI, AzureOpenAI

# Import HybridAgentContextManager for enhanced routing analytics and dual persistence
from .hybrid_agent_context import HybridAgentContextManager

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

# Agent Framework imports
try:
    from agent_framework import ChatAgent
    from agent_framework.azure import AzureAIAgentClient
    from azure.ai.projects.aio import AIProjectClient
    from azure.identity.aio import DefaultAzureCredential
    AGENT_FRAMEWORK_AVAILABLE = True
except ImportError:
    AGENT_FRAMEWORK_AVAILABLE = False
    print("⚠️ Agent Framework not available. Install with: pip install agent-framework-azure-ai")


@dataclass
class HybridAgentRouterConfig:
    """Configuration for the hybrid router with Agent Framework."""
    # Local model configuration (Foundry Local)
    local_endpoint: Optional[str] = None
    local_model_name: Optional[str] = None
    local_model_id: Optional[str] = None
    
    # Azure API Management configuration
    apim_endpoint: Optional[str] = None
    apim_key: Optional[str] = None
    apim_deployment_id: Optional[str] = None
    
    # Agent Framework configuration
    agent_project_endpoint: Optional[str] = None
    agent_model_deployment: Optional[str] = None
    agent_default_instructions: str = "You are a helpful AI assistant specialized in enterprise solutions and hybrid AI systems."
    
    # Azure OpenAI fallback configuration
    azure_endpoint: Optional[str] = None
    azure_key: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_api_version: str = "2024-12-01-preview"
    
    # ML router configuration
    bert_model_path: Optional[str] = None
    phi_model_path: Optional[str] = None
    ml_confidence_threshold: float = 0.7
    
    # Routing thresholds
    complexity_threshold: int = 5  # Score >= 5 routes to cloud
    word_count_threshold: int = 20  # Words >= 20 suggests complexity
    enterprise_keyword_threshold: int = 2  # Number of enterprise keywords
    
    # Context management
    max_context_length: int = 15
    enable_context_routing: bool = True


class HybridAgentRouter:
    """
    Two-tier hybrid router: Local (Foundry Local) → Azure Cloud (APIM + Agent Framework).
    
    Provides intelligent routing with ML-powered query analysis, enterprise-grade APIM 
    integration, and modern Agent Framework for complex reasoning tasks.
    """
    
    def __init__(self, config: HybridAgentRouterConfig, session_id: str = None):
        """Initialize the hybrid router with Agent Framework."""
        self.config = config
        self.routing_history = []
        
        # Initialize hybrid agent context manager with enhanced analytics
        self.context_manager = HybridAgentContextManager(
            session_id=session_id or f"agent_session_{int(time.time())}"
        )
        
        # Initialize ML routers
        self.bert_router = None
        self.phi_router = None
        self._init_ml_routers()
        
        # Initialize clients
        self.local_client = None
        self.apim_client = None
        self.azure_client = None
        
        # Agent Framework components
        self.agent_manager = None
        self.persistent_agents = {}
        
        self._init_clients()
        self._init_agent_framework()
    
    def _init_ml_routers(self):
        """Initialize ML-based routers if available."""
        # Initialize BERT router (preferred)
        if BERT_AVAILABLE and self.config.bert_model_path:
            try:
                bert_config = BertRouterConfig(
                    model_path=self.config.bert_model_path,
                    confidence_threshold=self.config.ml_confidence_threshold
                )
                self.bert_router = BertQueryRouter(bert_config)
                print("✅ BERT Router initialized")
            except Exception as e:
                print(f"⚠️ BERT Router initialization failed: {e}")
        
        # Initialize PHI router (fallback)
        if PHI_AVAILABLE and self.config.phi_model_path and self.bert_router is None:
            try:
                phi_config = PhiRouterConfig(
                    model_path=self.config.phi_model_path,
                    confidence_threshold=self.config.ml_confidence_threshold
                )
                self.phi_router = PhiQueryRouter(phi_config)
                print("✅ PHI Router initialized")
            except Exception as e:
                print(f"⚠️ PHI Router initialization failed: {e}")
    
    def _init_clients(self):
        """Initialize API clients for different routing targets."""
        # Local client (Foundry Local)
        if self.config.local_endpoint:
            try:
                # Ensure endpoint has /v1 suffix for OpenAI compatibility
                local_url = self.config.local_endpoint
                if not local_url.endswith('/v1'):
                    local_url = local_url.rstrip('/') + '/v1'
                
                self.local_client = OpenAI(
                    base_url=local_url,
                    api_key="not-needed"
                )
                print(f"✅ Local client initialized: {local_url}")
            except Exception as e:
                print(f"⚠️ Local client initialization failed: {e}")
        
        # APIM client
        if self.config.apim_endpoint and self.config.apim_key:
            try:
                self.apim_client = AzureOpenAI(
                    azure_endpoint=self.config.apim_endpoint,
                    api_key=self.config.apim_key,
                    api_version="2024-02-01"
                )
                print(f"✅ APIM client initialized: {self.config.apim_endpoint}")
            except Exception as e:
                print(f"⚠️ APIM client initialization failed: {e}")
        
        # Azure OpenAI fallback
        if self.config.azure_endpoint and self.config.azure_key:
            try:
                self.azure_client = AzureOpenAI(
                    api_key=self.config.azure_key,
                    api_version=self.config.azure_api_version,
                    azure_endpoint=self.config.azure_endpoint
                )
                print(f"✅ Azure OpenAI client initialized: {self.config.azure_endpoint}")
            except Exception as e:
                print(f"⚠️ Azure OpenAI client initialization failed: {e}")
    
    def _init_agent_framework(self):
        """Initialize Agent Framework if available."""
        if not AGENT_FRAMEWORK_AVAILABLE:
            print("⚠️ Agent Framework not available - cloud routing limited to APIM/Azure OpenAI")
            return
        
        if not self.config.agent_project_endpoint:
            print("⚠️ Agent project endpoint not configured")
            return
        
        try:
            print(f"✅ Agent Framework available for endpoint: {self.config.agent_project_endpoint}")
            print(f"   Model deployment: {self.config.agent_model_deployment}")
        except Exception as e:
            print(f"⚠️ Agent Framework initialization warning: {e}")
    
    def analyze_query_for_routing(self, query: str) -> Dict:
        """Enhanced query analysis using ML routers for two-tier hybrid routing."""
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
            'router_used': 'pattern_based',
            'reasoning': '',
            'ml_prediction': None,
            'ml_confidence': 0.0,
            'route_to': 'local'  # default
        }
        
        # Use BERT router if available (preferred)
        if self.bert_router is not None:
            try:
                prediction = self.bert_router.route_query(query)
                # route_query returns (route_name, reason, details_dict)
                analysis['ml_prediction'] = prediction[0] if isinstance(prediction, tuple) else prediction.get('route', 'local')
                analysis['ml_confidence'] = prediction[2].get('confidence', 0.0) if isinstance(prediction, tuple) and len(prediction) > 2 else prediction.get('confidence', 0.0)
                analysis['router_used'] = 'bert'
                analysis['reasoning'] = f"BERT predicted '{analysis['ml_prediction']}' with {analysis['ml_confidence']:.2%} confidence"
                
                # High-confidence BERT predictions override pattern-based
                if analysis['ml_confidence'] > self.config.ml_confidence_threshold:
                    analysis['route_to'] = analysis['ml_prediction']
                    return analysis
            except Exception as e:
                print(f"⚠️ BERT router error: {e}")
                analysis['router_used'] = 'bert_fallback'
        
        # Use PHI router if BERT unavailable
        elif self.phi_router is not None:
            try:
                prediction = self.phi_router.predict_route(query)
                analysis['ml_prediction'] = prediction['route']
                analysis['ml_confidence'] = prediction['confidence']
                analysis['router_used'] = 'phi'
                analysis['reasoning'] = f"PHI predicted '{prediction['route']}' with {prediction['confidence']:.2%} confidence"
                
                if prediction['confidence'] > self.config.ml_confidence_threshold:
                    analysis['route_to'] = prediction['route']
                    return analysis
            except Exception as e:
                print(f"⚠️ PHI router error: {e}")
                analysis['router_used'] = 'phi_fallback'
        
        # Fallback to pattern-based analysis
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
                           'compare', 'evaluate', 'strategy', 'plan', 'implications', 'assessment']
        creative_keywords = ['write a', 'create a', 'compose', 'design', 'brainstorm', 'imagine']
        enterprise_keywords = ['business', 'enterprise', 'production', 'scalable', 'architecture',
                              'compliance', 'security', 'deployment', 'infrastructure', 'roi',
                              'financial', 'strategic', 'organizational']
        
        # Apply pattern matching
        for pattern in greeting_patterns:
            if re.match(pattern, query_lower):
                analysis['is_greeting'] = True
        
        for pattern in simple_patterns:
            if re.match(pattern, query_lower):
                analysis['is_simple_question'] = True
        
        for pattern in calc_patterns:
            if re.search(pattern, query_lower):
                analysis['is_calculation'] = True
        
        # Check for complex keywords
        analysis['requires_analysis'] = any(kw in query_lower for kw in complex_keywords)
        analysis['requires_creativity'] = any(kw in query_lower for kw in creative_keywords)
        
        enterprise_count = sum(1 for kw in enterprise_keywords if kw in query_lower)
        analysis['is_enterprise_query'] = enterprise_count >= self.config.enterprise_keyword_threshold
        
        analysis['is_conversational'] = any(word in query_lower for word in 
                                          ['discuss', 'conversation', 'talk about', 'tell me about', 'chat'])
        
        # Calculate complexity score
        score = 0
        if analysis['word_count'] > self.config.word_count_threshold:
            score += 2
        if analysis['word_count'] > 50:
            score += 2
        if analysis['requires_analysis']:
            score += 3
        if analysis['requires_creativity']:
            score += 2
        if analysis['is_enterprise_query']:
            score += 3
        if analysis['is_conversational']:
            score += 1
        
        analysis['complexity_score'] = score
        
        # Routing decision based on complexity
        if (analysis['is_greeting'] or analysis['is_calculation'] or 
            (analysis['is_simple_question'] and analysis['word_count'] <= 10)):
            analysis['route_to'] = 'local'
            analysis['reasoning'] = 'Simple query suitable for local processing'
        elif score >= self.config.complexity_threshold:
            analysis['route_to'] = 'cloud'
            analysis['reasoning'] = f'Complex query (score={score}) requires cloud processing'
        else:
            analysis['route_to'] = 'local'
            analysis['reasoning'] = f'Moderate complexity (score={score}) suitable for local'
    
    def route_query_intelligently(self, query: str) -> Tuple[str, str, int]:
        """
        Determine optimal routing target for two-tier hybrid system.
        
        Returns:
            (target, reasoning, priority) where target is 'local' or 'cloud'
        """
        analysis = self.analyze_query_for_routing(query)
        
        # Priority 1: Local (Foundry Local) for simple, fast queries
        if analysis['route_to'] == 'local':
            return 'local', analysis['reasoning'], 1
        
        # Priority 2: Cloud (APIM/Agent Framework) for complex queries
        if analysis['route_to'] == 'cloud':
            return 'cloud', analysis['reasoning'], 2
        
        # Default to local
        return 'local', 'Default routing to local', 1
    
    def query_local_model(self, prompt: str, max_tokens: int = 200) -> Tuple[str, float, bool]:
        """Query local Foundry Local model."""
        if not self.local_client:
            print("⚠️ Local client not initialized")
            return "Local model not available", 0, False
        
        try:
            start_time = time.time()
            response = self.local_client.chat.completions.create(
                model=self.config.local_model_id or self.config.local_model_name or "local-model",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            response_time = time.time() - start_time
            print(f"✅ Local model response in {response_time:.3f}s")
            return response.choices[0].message.content, response_time, True
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Local model error: {error_msg}")
            if "connection" in error_msg.lower():
                print(f"💡 Check if Foundry Local is running at {self.config.local_endpoint}")
            return f"Local model error: {error_msg}", 0, False
    
    def query_apim_router(self, prompt: str, max_tokens: int = 500) -> Tuple[str, float, bool]:
        """Query through APIM Model Router for cloud routing."""
        if not self.apim_client:
            return "APIM not available", 0, False
        
        try:
            start_time = time.time()
            response = self.apim_client.chat.completions.create(
                model=self.config.apim_deployment_id or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            response_time = time.time() - start_time
            return response.choices[0].message.content, response_time, True
        except Exception as e:
            return f"APIM error: {str(e)}", 0, False
    
    async def query_agent_framework_async(self, prompt: str, instructions: str = None) -> Tuple[str, float, bool]:
        """Query using Agent Framework (async) for complex reasoning."""
        if not AGENT_FRAMEWORK_AVAILABLE:
            return "Agent Framework not available", 0, False
        
        if not self.config.agent_project_endpoint:
            return "Agent project endpoint not configured", 0, False
        
        try:
            start_time = time.time()
            
            agent_instructions = instructions or self.config.agent_default_instructions
            
            async with (
                DefaultAzureCredential() as credential,
                AzureAIAgentClient(
                    project_endpoint=self.config.agent_project_endpoint,
                    model_deployment_name=self.config.agent_model_deployment,
                    async_credential=credential,
                    agent_name="HybridRouterAgent"
                ).create_agent(
                    instructions=agent_instructions
                ) as agent,
            ):
                result = await agent.run(prompt)
                response_time = time.time() - start_time
                return result.text, response_time, True
                
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Agent Framework error: {error_msg}")
            
            # Provide helpful error messages
            if "authentication" in error_msg.lower():
                print("💡 Run: az login")
            elif "endpoint" in error_msg.lower():
                print("💡 Check agent_project_endpoint configuration")
            
            return f"Agent Framework error: {error_msg}", 0, False
    
    def query_agent_framework(self, prompt: str, instructions: str = None) -> Tuple[str, float, bool]:
        """Query using Agent Framework (sync wrapper)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a new one
                import nest_asyncio
                nest_asyncio.apply()
            return loop.run_until_complete(self.query_agent_framework_async(prompt, instructions))
        except Exception as e:
            # Fallback: create new event loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.query_agent_framework_async(prompt, instructions))
                loop.close()
                return result
            except Exception as e2:
                return f"Agent Framework sync error: {str(e2)}", 0, False
    
    def query_azure_direct(self, prompt: str, max_tokens: int = 500) -> Tuple[str, float, bool]:
        """Query Azure OpenAI directly (fallback)."""
        if not self.azure_client:
            return "Azure OpenAI not available", 0, False
        
        try:
            start_time = time.time()
            response = self.azure_client.chat.completions.create(
                model=self.config.azure_deployment or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            response_time = time.time() - start_time
            return response.choices[0].message.content, response_time, True
        except Exception as e:
            return f"Azure OpenAI error: {str(e)}", 0, False
    
    def route(self, query: str, show_reasoning: bool = False) -> str:
        """
        Process query through two-tier hybrid routing system (sync).
        
        Args:
            query: User query to route
            show_reasoning: Whether to include routing reasoning in response
            
        Returns:
            Formatted response with source indication
        """
        # Determine optimal routing target
        target, reason, priority = self.route_query_intelligently(query)
        
        response = ""
        response_time = 0
        success = False
        actual_source = target
        
        # Execute routing decision with fallback chain
        if target == 'local':
            # Route to local with cloud fallback
            response, response_time, success = self.query_local_model(query)
            
            if not success:
                print(f"⚠️ Local failed, trying APIM...")
                response, response_time, success = self.query_apim_router(query)
                actual_source = 'apim-fallback'
                
                if not success and AGENT_FRAMEWORK_AVAILABLE:
                    print(f"⚠️ APIM failed, trying Agent Framework...")
                    response, response_time, success = self.query_agent_framework(query)
                    actual_source = 'agent-fallback'
                
                if not success:
                    print(f"⚠️ Agent failed, trying Azure direct...")
                    response, response_time, success = self.query_azure_direct(query)
                    actual_source = 'azure-fallback'
        
        elif target == 'cloud':
            # Try Agent Framework first for complex queries
            if AGENT_FRAMEWORK_AVAILABLE:
                response, response_time, success = self.query_agent_framework(query)
                actual_source = 'agent'
                
                if not success:
                    print(f"⚠️ Agent failed, trying APIM...")
                    response, response_time, success = self.query_apim_router(query)
                    actual_source = 'apim-fallback'
            else:
                # No Agent Framework, use APIM
                response, response_time, success = self.query_apim_router(query)
                actual_source = 'apim'
            
            # Final fallbacks
            if not success:
                print(f"⚠️ APIM failed, trying Azure direct...")
                response, response_time, success = self.query_azure_direct(query)
                actual_source = 'azure-fallback'
            
            if not success:
                print(f"⚠️ Azure failed, trying local...")
                response, response_time, success = self.query_local_model(query)
                actual_source = 'local-fallback'
        
        # Format response with source indication
        source_tags = {
            'local': '[LOCAL]',
            'apim': '[APIM]',
            'agent': '[AGENT]',
            'apim-fallback': '[APIM*]',
            'agent-fallback': '[AGENT*]',
            'azure-fallback': '[AZURE*]',
            'local-fallback': '[LOCAL*]'
        }
        
        if success:
            tag = source_tags.get(actual_source, '[UNKNOWN]')
            formatted_response = f"{tag} {response}"
            if show_reasoning:
                formatted_response = f"{formatted_response}\n\n[Routing: {reason}]"
        else:
            formatted_response = f"[ERROR] All routing attempts failed: {response}"
        
        # Record for statistics
        self.routing_history.append({
            'query': query,
            'target': target,
            'actual_source': actual_source,
            'response_time': response_time,
            'success': success,
            'timestamp': time.time()
        })
        
        return formatted_response
    
    async def route_async(self, query: str, use_context: bool = True, show_reasoning: bool = False) -> Dict[str, Any]:
        """
        Process query through two-tier hybrid routing system (async).
        
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
        if use_context and self.config.enable_context_routing:
            context_messages = self.context_manager.get_recent_exchanges(5)
        
        # Prepare query with context
        query_with_context = query
        if context_messages:
            context_str = "\n".join([
                f"User: {ex.get('user_message', '')}\nAssistant: {ex.get('response', '')}" 
                for ex in context_messages
            ])
            query_with_context = f"Previous context:\n{context_str}\n\nCurrent query: {query}"
        
        # Determine optimal routing target
        target, reason, priority = self.route_query_intelligently(query)
        analysis = self.analyze_query_for_routing(query)
        
        response = ""
        response_time = 0
        success = False
        actual_source = target
        
        # Execute routing decision with fallback chain
        if target == 'local':
            # Route to local with cloud fallback
            print(f"🏠 Routing to LOCAL model (endpoint: {self.config.local_endpoint})")
            response, response_time, success = self.query_local_model(query_with_context)
            
            if not success:
                print(f"⚠️ Local model failed, trying APIM fallback...")
                response, response_time, success = self.query_apim_router(query_with_context)
                actual_source = 'apim-fallback'
                
                if not success and AGENT_FRAMEWORK_AVAILABLE:
                    print(f"⚠️ APIM failed, trying Agent Framework fallback...")
                    response, response_time, success = await self.query_agent_framework_async(query_with_context)
                    actual_source = 'agent-fallback'
                
                if not success:
                    print(f"⚠️ Agent Framework failed, trying Azure OpenAI fallback...")
                    response, response_time, success = self.query_azure_direct(query_with_context)
                    actual_source = 'azure-fallback'
        
        elif target == 'cloud':
            # Try Agent Framework first for complex queries
            if AGENT_FRAMEWORK_AVAILABLE:
                response, response_time, success = await self.query_agent_framework_async(query_with_context)
                actual_source = 'agent'
                
                if not success:
                    response, response_time, success = self.query_apim_router(query_with_context)
                    actual_source = 'apim-fallback'
            else:
                # No Agent Framework, use APIM
                response, response_time, success = self.query_apim_router(query_with_context)
                actual_source = 'apim'
            
            # Final fallbacks
            if not success:
                response, response_time, success = self.query_azure_direct(query_with_context)
                actual_source = 'azure-fallback'
            
            if not success:
                response, response_time, success = self.query_local_model(query_with_context)
                actual_source = 'local-fallback'
        
        total_time = time.time() - start_time
        
        # Clean response content (remove source tags for storage)
        clean_response = response
        source_tags = ['[LOCAL]', '[APIM]', '[AGENT]', '[APIM*]', '[AGENT*]', '[AZURE*]', '[LOCAL*]', '[ERROR]']
        for tag in source_tags:
            if clean_response.startswith(tag):
                clean_response = clean_response[len(tag):].strip()
        
        # Create metadata
        metadata = {
            'strategy': target,
            'context_used': use_context and len(context_messages) > 0,
            'routing_info': {
                'target': target,
                'actual_source': actual_source,
                'reason': reason,
                'priority': priority,
                'analysis': analysis,
                'confidence': analysis.get('ml_confidence', 0.0),
                'router_used': analysis.get('router_used', 'pattern_based')
            },
            'response_time': total_time,
            'success': success,
            'context_length': len(context_messages),
            'query_length': len(query)
        }
        
        # Add exchange to conversation context using HybridAgentContextManager
        if use_context:
            self.context_manager.add_exchange_generic(
                prompt=query,
                response=clean_response,
                source=actual_source,
                response_time=total_time,
                metadata=metadata
            )
        
        # Record for statistics
        self.routing_history.append({
            'query': query,
            'target': target,
            'actual_source': actual_source,
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
            'agent': '[AGENT]',
            'apim-fallback': '[APIM*]',
            'agent-fallback': '[AGENT*]',
            'azure-fallback': '[AZURE*]',
            'local-fallback': '[LOCAL*]'
        }
        
        if success:
            tag = source_tags_dict.get(actual_source, '[UNKNOWN]')
            formatted_response = f"{tag} {clean_response}"
            if show_reasoning:
                formatted_response = f"{formatted_response}\n\n[Routing: {reason}]"
        else:
            formatted_response = f"[ERROR] All routing attempts failed: {clean_response}"
        
        return {
            'response': clean_response,
            'formatted_response': formatted_response,
            'source': actual_source,
            'responseTime': total_time,
            'metadata': metadata,
            'success': success
        }
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        if not self.routing_history:
            return {'message': 'No routing history available'}
        
        total = len(self.routing_history)
        successful = sum(1 for h in self.routing_history if h['success'])
        
        # Count by target
        local_routes = sum(1 for h in self.routing_history if 'local' in h['actual_source'])
        cloud_routes = sum(1 for h in self.routing_history if h['actual_source'] in ['apim', 'agent', 'azure'])
        fallback_routes = sum(1 for h in self.routing_history if 'fallback' in h['actual_source'])
        
        # Performance metrics
        successful_times = [h['response_time'] for h in self.routing_history if h['success']]
        avg_time = sum(successful_times) / max(len(successful_times), 1)
        
        return {
            'total_queries': total,
            'successful_queries': successful,
            'success_rate': successful / total * 100,
            'routing_distribution': {
                'local': local_routes,
                'cloud': cloud_routes,
                'fallbacks': fallback_routes
            },
            'routing_percentages': {
                'local': local_routes / total * 100,
                'cloud': cloud_routes / total * 100,
                'fallbacks': fallback_routes / total * 100
            },
            'performance': {
                'average_response_time': avg_time,
                'fastest_response': min(successful_times) if successful_times else 0,
                'slowest_response': max(successful_times) if successful_times else 0
            }
        }
    
    def get_system_capabilities(self) -> Dict:
        """Get system capabilities."""
        return {
            'available_targets': {
                'local_foundry': self.local_client is not None,
                'apim_router': self.apim_client is not None,
                'agent_framework': AGENT_FRAMEWORK_AVAILABLE and self.config.agent_project_endpoint is not None,
                'azure_direct': self.azure_client is not None
            },
            'ml_routers': {
                'bert_router': self.bert_router is not None,
                'phi_router': self.phi_router is not None
            },
            'agent_framework': {
                'available': AGENT_FRAMEWORK_AVAILABLE,
                'endpoint': self.config.agent_project_endpoint,
                'model': self.config.agent_model_deployment
            },
            'routing_strategy': 'Two-tier: Local (Foundry Local) → Cloud (APIM/Agent Framework)',
            'context_management': {
                'enabled': self.config.enable_context_routing,
                'session_id': self.context_manager.session_id,
                'max_history': self.config.max_context_length
            }
        }
    
    # Conversation Context Management Methods
    def get_conversation_history(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation exchanges."""
        return self.context_manager.get_recent_exchanges(count)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        context_summary = self.context_manager.get_session_summary()
        routing_stats = self.get_routing_stats()
        
        return {
            'session_info': context_summary.get('session_info', {}),
            'routing_stats': routing_stats,
            'conversation_flow': context_summary.get('conversation_flow', []),
            'context_stats': self.context_manager.get_conversation_summary()
        }
    
    def clear_conversation_context(self):
        """Clear conversation history."""
        self.context_manager.clear_conversation()
        print(f"🧹 Conversation context cleared for session {self.context_manager.session_id}")
    
    def export_conversation_history(self, filename: str = None) -> str:
        """Export conversation history to JSON file."""
        return self.context_manager.export_conversation(filename)


def create_hybrid_agent_router_from_env(session_id: str = None) -> HybridAgentRouter:
    """Create hybrid router with Agent Framework using environment variables."""
    config = HybridAgentRouterConfig(
        # Local configuration
        local_endpoint=os.environ.get("LOCAL_MODEL_ENDPOINT"),
        local_model_name=os.environ.get("LOCAL_MODEL_NAME"),
        local_model_id=os.environ.get("LOCAL_MODEL_ID"),
        
        # APIM configuration
        apim_endpoint=os.environ.get("APIM_ENDPOINT"),
        apim_key=os.environ.get("APIM_API_KEY"),
        apim_deployment_id=os.environ.get("AZURE_APIM_DEPLOYMENT_ID"),
        
        # Agent Framework configuration
        agent_project_endpoint=os.environ.get("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT"),
        agent_model_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini"),
        
        # Azure OpenAI fallback
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        azure_key=os.environ.get("AZURE_OPENAI_KEY"),
        azure_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME"),
        azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        
        # ML routers
        bert_model_path=os.environ.get("BERT_MODEL_FULLPATH"),
        phi_model_path=os.environ.get("PHI_MODEL_FULLPATH"),
        ml_confidence_threshold=float(os.environ.get("ML_CONFIDENCE_THRESHOLD", "0.7"))
    )
    
    return HybridAgentRouter(config, session_id=session_id)


# Test function
async def test_agent_router_integration():
    """Test the Agent Framework integration."""
    print("🧪 Testing HybridAgentRouter with Agent Framework...")
    
    # Create minimal config for testing
    config = HybridAgentRouterConfig(
        local_endpoint="http://localhost:1234",
        local_model_id="test-model",
        agent_project_endpoint=os.environ.get("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT"),
        agent_model_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4o-mini")
    )
    
    router = HybridAgentRouter(config, session_id="test_session")
    
    print(f"✅ Router created with session: {router.context_manager.session_id}")
    
    # Test routing analysis
    test_queries = [
        "Hello!",
        "Analyze this complex business strategy document...",
        "What is 2 + 2?"
    ]
    
    for query in test_queries:
        analysis = router.analyze_query_for_routing(query)
        print(f"\n📝 Query: {query}")
        print(f"   Route: {analysis['route_to']} | Score: {analysis['complexity_score']} | Router: {analysis['router_used']}")
    
    # Test capabilities
    capabilities = router.get_system_capabilities()
    print(f"\n🔧 System Capabilities:")
    for key, value in capabilities.items():
        print(f"   {key}: {value}")
    
    print("\n🎉 Agent Framework integration test completed!")


if __name__ == "__main__":
    asyncio.run(test_agent_router_integration())
