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

# Add project root directory for imports FIRST
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment from parent directory (main .env with all configs)
parent_env = os.path.join(project_root, '.env')
if os.path.exists(parent_env):
    load_dotenv(parent_env, override=True)
    print(f"[OK] Loaded environment from: {parent_env}")
else:
    load_dotenv()
    print(f"[WARN] Loading .env from current directory")

# Verify critical environment variables
print(f"[INFO] Environment Check:")
print(f"   LOCAL_MODEL_ENDPOINT: {os.environ.get('LOCAL_MODEL_ENDPOINT', 'NOT SET')}")
print(f"   APIM_ENDPOINT: {os.environ.get('APIM_ENDPOINT', 'NOT SET')}")
print(f"   AZURE_AI_FOUNDRY_PROJECT_ENDPOINT: {os.environ.get('AZURE_AI_FOUNDRY_PROJECT_ENDPOINT', 'NOT SET')}")
print(f"   BERT_MODEL_FULLPATH: {os.environ.get('BERT_MODEL_FULLPATH', 'NOT SET')}")

# Suppress warnings that might clutter the logs
warnings.filterwarnings('ignore')

# Try to import hybrid router (HybridAgentContextManager is now part of the router)
try:
    from modules.hybrid_router_agent_framework import HybridAgentRouter, HybridAgentRouterConfig, create_hybrid_agent_router_from_env
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

class EnhancedHybridRouterAPI:
    """Enhanced hybrid router API that mirrors streamlit_multiturn_demo.py functionality."""
    
    def __init__(self):
        self.router = None
        self.routing_stats = {"local": 0, "apim": 0, "foundry": 0, "azure": 0, "mock": 0, "error": 0}
        self.performance_history = []
        self.available_routers = {'hybrid': True, 'rule_based': False, 'bert': False, 'phi': False}
        self.selected_strategy = 'hybrid'
        # Routers have their own context managers - we'll access via router.context_manager
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
        """Initialize the Agent Framework hybrid router with error handling."""
        if router_available:
            try:
                with self.suppress_output():
                    # Create router with Agent Framework integration
                    session_id = f"react_api_{int(time.time())}"
                    self.router = create_hybrid_agent_router_from_env(session_id=session_id)
                
                if self.router:
                    self.available_routers['hybrid'] = True
                    logger.info(f"Agent Framework hybrid router initialized - Session: {self.router.context_manager.session_id}")
                    return True
                else:
                    self.available_routers['hybrid'] = False
                    logger.warning("Hybrid router initialization returned None")
                    return False
                    
            except Exception as e:
                logger.error(f"Agent Framework hybrid router initialization failed: {e}")
                self.available_routers['hybrid'] = False
                return False
        
        self.available_routers['hybrid'] = False
        logger.warning("Agent Framework hybrid router modules not available")
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
            # Use absolute path relative to project root
            bert_model_path = os.path.join(project_root, "notebooks", "mobilbert_query_router_trained")
            bert_config = BertRouterConfig(
                model_path=bert_model_path,
                confidence_threshold=0.7
            )
            self.bert_router = BertQueryRouter(bert_config)
            self.available_routers['bert'] = True
            logger.info(f"BERT router initialized successfully from {bert_model_path}")
        except Exception as e:
            logger.error(f"BERT router initialization failed: {e}")
            self.bert_router = None
            self.available_routers['bert'] = False
        
        # Initialize PHI router
        try:
            # Use absolute path relative to project root
            phi_model_path = os.path.join(project_root, "notebooks", "phi_router_model")
            phi_config = PhiRouterConfig(model_path=phi_model_path)
            self.phi_router = PhiQueryRouter(phi_config)
            self.available_routers['phi'] = True
            logger.info(f"PHI router initialized successfully from {phi_model_path}")
        except Exception as e:
            logger.error(f"PHI router initialization failed: {e}")
            self.phi_router = None
            self.available_routers['phi'] = False
    
    async def route_with_selected_strategy_async(self, query: str, strategy: str, session_id: str) -> Tuple[str, str, float, Dict]:
        """Route query using selected strategy with Agent Framework async support."""
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
            # Use Agent Framework async routing with context (router manages its own context)
            has_context = len(self.router.context_manager.routing_metadata) > 0
            
            # Route using Agent Framework (async) - properly await the result
            result = await self.router.route_async(
                query=query,
                use_context=has_context,
                show_reasoning=False
            )
            
            response = result['response']
            actual_source = result['source']
            response_time = result['responseTime']
            
            # Extract metadata from Agent Framework result
            routing_metadata = result.get('metadata', {})
            routing_info = routing_metadata.get('routing_info', {})
            
            metadata = {
                'strategy': strategy,
                'target': routing_info.get('target', actual_source),
                'confidence': routing_info.get('confidence', 0.0),
                'reason': routing_info.get('reason', 'Agent Framework routing'),
                'source': actual_source,
                'response_time': response_time,
                'success': result.get('success', True),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'analysis': routing_info.get('analysis', {}),
                'context_used': routing_metadata.get('context_used', False),
                'router_used': routing_info.get('router_used', 'agent_framework')
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
        has_context = False  # No context available without router
        
        if is_followup and has_context:
            # Context not available in mock mode
            response = f"Mock response (no context available): {query[:50]}..."
            return response, "mock"
        
        # Standard routing logic with context awareness
        if any(word in query_lower for word in ["hello", "hi", "start", "begin"]):
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
    """Route a query using the specified strategy with Agent Framework."""
    try:
        session_id = request.session_id or f"session_{int(time.time())}"
        
        # Use async routing with Agent Framework
        response, source, response_time, metadata = await api_instance.route_with_selected_strategy_async(
            request.query, request.strategy, session_id
        )
        
        # Context is managed by the router's HybridAgentContextManager automatically
        # No need to manually add exchanges - the router.route_async() does this
        
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
        logger.error(f"Error routing query with Agent Framework: {e}")
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

@app.get("/api/health")
async def api_health_check():
    """Health check endpoint with /api prefix for React frontend."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/query", response_model=QueryResponse)
async def api_route_query(request: QueryRequest):
    """Route a query using the specified strategy (React /api/* pattern)."""
    return await route_query(request)

@app.get("/api/system-status", response_model=SystemStatusResponse)
async def api_get_system_status():
    """Get system status and available routers (React /api/* pattern)."""
    return await get_system_status()

@app.get("/api/capabilities")
async def api_get_capabilities():
    """Get system capabilities (React /api/* pattern)."""
    return {
        "availableRouters": api_instance.available_routers,
        "systemHealth": api_instance.get_system_status()['systemHealth'],
        "features": {
            "multiTurnConversation": True,
            "contextManagement": True,
            "hybridRouting": api_instance.available_routers.get('hybrid', False),
            "bertRouting": api_instance.available_routers.get('bert', False),
            "phiRouting": api_instance.available_routers.get('phi', False)
        }
    }

@app.delete("/api/clear-context/{session_id}")
async def api_clear_context(session_id: str):
    """Clear conversation context for a session (React /api/* pattern)."""
    try:
        if api_instance.router and api_instance.router.context_manager:
            api_instance.router.context_manager.clear_conversation()
            return {"status": "success", "message": f"Context cleared for session {session_id}"}
        return {"status": "info", "message": "No router context available"}
    except Exception as e:
        logger.error(f"Error clearing context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversation-history/{session_id}")
async def api_get_conversation_history(session_id: str):
    """Get conversation history for a session (React /api/* pattern)."""
    try:
        if api_instance.router and api_instance.router.context_manager:
            # Get full conversation messages using the dedicated method
            messages = api_instance.router.context_manager.get_conversation_messages(count=50)
            summary = api_instance.router.context_manager.get_routing_summary()
            
            return {
                "session_id": session_id,
                "exchanges": messages,
                "stats": summary
            }
        return {"session_id": session_id, "exchanges": [], "stats": {}}
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversation-insights/{session_id}")
async def api_get_conversation_insights(session_id: str):
    """Get conversation insights and analytics (React /api/* pattern)."""
    try:
        if not api_instance.router or not api_instance.router.context_manager:
            return {"session_id": session_id, "insights": {}, "message": "No router available"}
        
        summary = api_instance.router.context_manager.get_routing_summary()
        
        return {
            "session_id": session_id,
            "insights": {
                "total_exchanges": summary.get('total_exchanges', 0),
                "model_switches": summary.get('model_switches', 0),
                "routing_distribution": summary.get('routing_distribution', {}),
                "performance_metrics": summary.get('performance_metrics', {}),
                "has_agent_thread": summary.get('has_agent_thread', False)
            }
        }
    except Exception as e:
        logger.error(f"Error getting conversation insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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