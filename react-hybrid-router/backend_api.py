#!/usr/bin/env python3
"""
Enhanced FastAPI backend for React hybrid router application.
Mirrors the functionality of streamlit_multiturn_demo.py.
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import warnings
from contextlib import contextmanager
import io

# Load environment
load_dotenv()

# Add project root directory for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress warnings that might clutter the logs
warnings.filterwarnings('ignore')

# Try to import hybrid router and context manager
try:
    from modules.hybrid_router import HybridFoundryAPIMRouter, HybridRouterConfig, create_hybrid_router_from_env
    from modules.context_manager import ConversationManager, ConversationMessage, MessageRole, ModelSource
    router_available = True
except ImportError as e:
    router_available = False
    print(f"Warning: Hybrid router modules not available: {e}")

# Try to import additional routing modules
try:
    from modules.bert_router import BertQueryRouter, BertRouterConfig
    from modules.phi_router import PhiQueryRouter, PhiRouterConfig
    additional_routers_available = True
except ImportError as e:
    additional_routers_available = False
    print(f"Warning: Additional router modules not available: {e}")

# FastAPI app setup
app = FastAPI(
    title="Hybrid AI Router API",
    description="Enhanced backend for React hybrid router application with multi-turn conversation support",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    strategy: str = "hybrid"
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    source: str
    responseTime: float
    metadata: Dict[str, Any]

class SystemStatusResponse(BaseModel):
    availableRouters: Dict[str, bool]
    systemHealth: str  # 'healthy' | 'degraded' | 'error'

class ConversationContext:
    """Enhanced conversation context manager for the API."""
    
    def __init__(self, max_exchanges: int = 15):
        self.max_exchanges = max_exchanges
        self.sessions: Dict[str, List[Dict]] = {}
        self.session_stats: Dict[str, Dict] = {}
    
    def add_exchange(self, session_id: str, user_msg: str, ai_response: str, source: str, response_time: float, metadata: Optional[Dict] = None):
        """Add a conversation exchange with context preservation."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            self.session_stats[session_id] = {
                'start_time': datetime.now(),
                'total_exchanges': 0,
                'model_switches': 0,
                'last_model': None,
                'context_preserved': 0,
                'fallback_used': 0
            }
        
        exchange = {
            'timestamp': datetime.now(),
            'user_message': user_msg,
            'ai_response': ai_response,
            'source': source,
            'response_time': response_time,
            'exchange_number': len(self.sessions[session_id]) + 1,
            'metadata': metadata or {}
        }
        
        # Update session stats
        stats = self.session_stats[session_id]
        stats['total_exchanges'] += 1
        
        # Track model switches
        if stats['last_model'] and stats['last_model'] != source:
            stats['model_switches'] += 1
            exchange['model_switched'] = True
        else:
            exchange['model_switched'] = False
        
        stats['last_model'] = source
        
        # Add to history
        self.sessions[session_id].append(exchange)
        
        # Maintain max history length
        if len(self.sessions[session_id]) > self.max_exchanges:
            self.sessions[session_id].pop(0)
    
    def get_context_for_query(self, session_id: str, current_query: str) -> str:
        """Generate context-aware query with conversation history."""
        if session_id not in self.sessions or len(self.sessions[session_id]) == 0:
            return current_query
        
        # Build context summary
        recent_exchanges = self.sessions[session_id][-3:]  # Last 3 exchanges
        context_summary = []
        
        for ex in recent_exchanges:
            context_summary.append(f"User: {ex['user_message'][:50]}...")
            context_summary.append(f"AI: {ex['ai_response'][:50]}...")
        
        context_info = " | ".join(context_summary)
        return f"{current_query} [Context: {len(self.sessions[session_id])} exchanges: {context_info}]"
    
    def get_formatted_history(self, session_id: str) -> List[Dict]:
        """Get conversation history in OpenAI format."""
        if session_id not in self.sessions:
            return []
        
        messages = []
        for exchange in self.sessions[session_id][-10:]:  # Last 10 exchanges
            messages.append({"role": "user", "content": exchange['user_message']})
            messages.append({"role": "assistant", "content": exchange['ai_response']})
        return messages

class EnhancedHybridRouterAPI:
    """Enhanced hybrid router API that mirrors streamlit_multiturn_demo.py functionality."""
    
    def __init__(self):
        self.router = None
        self.context = ConversationContext()
        self.routing_stats = {"local": 0, "apim": 0, "foundry": 0, "azure": 0, "mock": 0, "error": 0}
        self.performance_history = []
        self.available_routers = {'hybrid': True, 'rule_based': False, 'bert': False, 'phi': False}
        self.selected_strategy = 'hybrid'
        self.init_router()
        self.init_additional_routers()
    
    @contextmanager
    def suppress_output(self):
        """Context manager to suppress stdout/stderr during router initialization."""
        with io.StringIO() as buf, io.StringIO() as err_buf:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = buf, err_buf
                yield
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
    
    def init_router(self):
        """Initialize the hybrid router with error handling."""
        if router_available:
            try:
                with self.suppress_output():
                    self.router = create_hybrid_router_from_env()
                
                if self.router:
                    self.available_routers['hybrid'] = True
                    logger.info("Hybrid router initialized successfully")
                    return True
                else:
                    self.available_routers['hybrid'] = False
                    logger.warning("Hybrid router initialization returned None")
                    return False
                    
            except Exception as e:
                logger.error(f"Hybrid router initialization failed: {e}")
                self.available_routers['hybrid'] = False
                return False
        
        self.available_routers['hybrid'] = False
        logger.warning("Hybrid router modules not available")
        return False
    
    def init_additional_routers(self):
        """Initialize additional routing strategies."""
        if not additional_routers_available:
            logger.warning("Additional router modules not available")
            return
        
        # Rule-based routing is now handled through hybrid router
        self.available_routers['rule_based'] = True
        
        # Initialize BERT router
        try:
            bert_config = BertRouterConfig(
                model_path="./notebooks/mobilbert_query_router_trained",
                confidence_threshold=0.7
            )
            self.bert_router = BertQueryRouter(bert_config)
            self.available_routers['bert'] = True
            logger.info("BERT router initialized successfully")
        except Exception as e:
            logger.error(f"BERT router initialization failed: {e}")
            self.bert_router = None
            self.available_routers['bert'] = False
        
        # Initialize PHI router
        try:
            phi_config = PhiRouterConfig(model_path="./notebooks/phi_router_model")
            self.phi_router = PhiQueryRouter(phi_config)
            self.available_routers['phi'] = True
            logger.info("PHI router initialized successfully")
        except Exception as e:
            logger.error(f"PHI router initialization failed: {e}")
            self.phi_router = None
            self.available_routers['phi'] = False
    
    def route_with_selected_strategy(self, query: str, strategy: str, session_id: str) -> Tuple[str, str, float, Dict]:
        """Route query using selected strategy and generate response."""
        start_time = time.time()
        
        if not self.router:
            # Generate mock response without router
            response_time = time.time() - start_time
            mock_response, source = self._generate_contextual_mock_response(query, session_id)
            
            metadata = {
                'strategy': 'mock',
                'target': source,
                'confidence': 0.5,
                'reason': 'Router not available',
                'source': source,
                'response_time': response_time,
                'success': True,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return mock_response, source, response_time, metadata
        
        try:
            # Get routing decision using enhanced strategy matching
            target, reason, confidence = self._route_with_strategy(query, strategy)
            
            # Get analysis information for router used details
            analysis = {}
            if self.router and strategy in ['hybrid', 'rule_based']:
                try:
                    analysis = self.router.analyze_query_for_hybrid_routing(query)
                except Exception:
                    analysis = {'router_used': strategy}
            else:
                analysis = {'router_used': strategy}
            
            # Generate context-aware query
            context_query = self.context.get_context_for_query(session_id, query)
            
            # Generate response using the router
            response = self.router.route(context_query, show_reasoning=False)
            
            # Parse response to extract source
            actual_source = target
            if response.startswith("["):
                end_bracket = response.find("]")
                if end_bracket != -1:
                    actual_source = response[1:end_bracket].lower()
                    response = response[end_bracket+1:].strip()
            
            response_time = time.time() - start_time
            
            metadata = {
                'strategy': strategy,
                'target': target,
                'confidence': confidence,
                'reason': reason,
                'source': actual_source,
                'response_time': response_time,
                'success': True,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'analysis': analysis,
                'context_used': session_id in self.context.sessions and len(self.context.sessions[session_id]) > 0
            }
            
            return response, actual_source, response_time, metadata
            
        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Error processing request: {str(e)}"
            
            metadata = {
                'strategy': 'error',
                'target': 'error',
                'confidence': 0.0,
                'reason': f'Routing error: {str(e)}',
                'source': 'error',
                'response_time': response_time,
                'success': False,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return error_msg, "error", response_time, metadata
    
    def _route_with_strategy(self, query: str, strategy: str) -> Tuple[str, str, float]:
        """Route query using selected strategy (matches streamlit logic)."""
        try:
            if strategy == 'hybrid':
                target, reason, priority = self.router.route_hybrid_query(query)
                analysis = self.router.analyze_query_for_hybrid_routing(query)
                confidence = analysis.get('ml_confidence', 0.75)
            elif strategy == 'rule_based':
                analysis = self.router.analyze_query_for_hybrid_routing(query)
                if analysis['is_greeting'] or analysis['is_calculation']:
                    target, reason = 'local', 'Simple query - rule based'
                else:
                    target, reason = 'cloud', 'Complex query - rule based'
                confidence = 0.85
            elif strategy == 'bert':
                if hasattr(self.router, 'bert_router') and self.router.bert_router:
                    target, reason, metadata = self.router.bert_router.route_query(query)
                    confidence = metadata.get('confidence', 0.75)
                else:
                    target, reason, confidence = 'cloud', 'BERT router not available', 0.5
            elif strategy == 'phi':
                if hasattr(self.router, 'phi_router') and self.router.phi_router:
                    target, reason, metadata = self.router.phi_router.route_query(query)
                    confidence = metadata.get('confidence', 0.75)
                else:
                    target, reason, confidence = 'cloud', 'PHI router not available', 0.5
            else:
                target, reason, confidence = 'cloud', 'Unknown strategy', 0.5
                
            return target, reason, confidence
            
        except Exception as e:
            return 'cloud', f'Routing error: {str(e)[:50]}', 0.5
    
    def _generate_contextual_mock_response(self, query: str, session_id: str) -> Tuple[str, str]:
        """Generate context-aware mock responses."""
        import random
        
        query_lower = query.lower()
        
        # Check for context-dependent queries
        is_followup = any(word in query_lower for word in ["what about", "and", "also", "furthermore", "additionally", "can you", "elaborate", "explain more"])
        has_context = session_id in self.context.sessions and len(self.context.sessions[session_id]) > 0
        
        if is_followup and has_context:
            last_exchange = self.context.sessions[session_id][-1]
            last_source = last_exchange['source']
            
            responses = [
                f"Following up on our previous discussion about '{last_exchange['user_message'][:30]}...', let me elaborate: {query[:50]}",
                f"Based on our conversation context, I can expand on that topic: {query[:50]}",
                f"Continuing from my previous {last_source} response, here's more detail: {query[:50]}"
            ]
            return random.choice(responses), last_source
        
        # Standard routing logic with context awareness
        if any(word in query_lower for word in ["hello", "hi", "start", "begin"]):
            if has_context:
                responses = ["Welcome back! I remember our previous conversation.", "Hello again! Ready to continue our discussion?"]
            else:
                responses = ["Hello! I'm ready to start our conversation.", "Hi there! Let's begin our chat."]
            source = "local"
        elif any(word in query_lower for word in ["enterprise", "business", "production", "scalable"]):
            responses = [
                "Enterprise analysis: This complex business query would be routed through APIM for optimal processing.",
                "Business context detected: APIM router would handle this enterprise-level request."
            ]
            source = "apim"
        elif any(word in query_lower for word in ["analyze", "complex", "detailed", "comprehensive", "reasoning"]):
            responses = [
                "Complex analysis requested: Foundry Agent would provide detailed reasoning and comprehensive insights.",
                "Advanced query detected: Azure AI Foundry would handle this sophisticated analysis."
            ]
            source = "foundry"
        else:
            responses = [
                "General query: Azure OpenAI would process this standard request with cloud capabilities.",
                "Standard processing: Cloud model would handle this general inquiry."
            ]
            source = "azure"
        
        return random.choice(responses), source
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and available routers."""
        # Determine system health
        available_count = sum(1 for available in self.available_routers.values() if available)
        total_count = len(self.available_routers)
        
        if available_count == 0:
            health = 'error'
        elif available_count < total_count:
            health = 'degraded'
        else:
            health = 'healthy'
        
        return {
            'availableRouters': self.available_routers,
            'systemHealth': health
        }

# Global API instance
api_instance = EnhancedHybridRouterAPI()

# API endpoints
@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status and available routers."""
    try:
        status = api_instance.get_system_status()
        return SystemStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/route", response_model=QueryResponse)
async def route_query(request: QueryRequest):
    """Route a query using the specified strategy."""
    try:
        session_id = request.session_id or f"session_{int(time.time())}"
        
        response, source, response_time, metadata = api_instance.route_with_selected_strategy(
            request.query, request.strategy, session_id
        )
        
        # Add to context
        api_instance.context.add_exchange(
            session_id, request.query, response, source, response_time, metadata
        )
        
        # Update routing stats
        if source in api_instance.routing_stats:
            api_instance.routing_stats[source] += 1
        
        return QueryResponse(
            response=response,
            source=source,
            responseTime=response_time,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error routing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/route/bert", response_model=QueryResponse)
async def route_query_bert(request: QueryRequest):
    """Route a query specifically using BERT strategy."""
    try:
        request.strategy = "bert"
        return await route_query(request)
    except Exception as e:
        logger.error(f"Error routing query with BERT: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/route/phi", response_model=QueryResponse)
async def route_query_phi(request: QueryRequest):
    """Route a query specifically using PHI strategy."""
    try:
        request.strategy = "phi"
        return await route_query(request)
    except Exception as e:
        logger.error(f"Error routing query with PHI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Hybrid AI Router API",
        "version": "1.0.0",
        "description": "Enhanced backend for React hybrid router application",
        "endpoints": {
            "/status": "Get system status and available routers",
            "/route": "Route queries using specified strategy",
            "/route/bert": "Route queries using BERT strategy",
            "/route/phi": "Route queries using PHI strategy",
            "/health": "Health check endpoint"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting Enhanced Hybrid AI Router API...")
    logger.info(f"Router available: {router_available}")
    logger.info(f"Additional routers available: {additional_routers_available}")
    logger.info(f"Available routers: {api_instance.available_routers}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )