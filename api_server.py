#!/usr/bin/env python3
"""
FastAPI Backend for Hybrid LLM Router
Provides REST API endpoints for the React frontend
"""

import os
import sys
import time
import json
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Import the hybrid router modules
try:
    from modules.hybrid_router import HybridFoundryAPIMRouter, HybridRouterConfig, create_hybrid_router_from_env
    from modules.azure_ai_manager import AzureAIManager
    from modules.context_manager import ConversationContextManager, ModelSource
    from modules.telemetry import TelemetryCollector
    router_available = True
except ImportError as e:
    print(f"Warning: Could not import router modules: {e}")
    router_available = False

# FastAPI app initialization
app = FastAPI(
    title="Hybrid LLM Router API",
    description="REST API for intelligent routing between local and cloud LLM models",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
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
    systemHealth: str

# Global instances
hybrid_router = None
azure_manager = None
session_contexts = {}  # Dictionary to store session-based ConversationContextManager instances
telemetry_manager = None

def initialize_services():
    """Initialize all the services needed for the hybrid router"""
    global hybrid_router, azure_manager, session_contexts, telemetry_manager
    
    try:
        # Initialize hybrid router
        if router_available:
            try:
                hybrid_router = create_hybrid_router_from_env()
                print("✅ Hybrid Router (HybridFoundryAPIMRouter) initialized")
            except Exception as e:
                print(f"⚠️ Hybrid Router initialization failed: {e}")
                hybrid_router = None
        
        # Initialize Azure AI Manager
        try:
            azure_manager = AzureAIManager()
            print("✅ Azure AI Manager initialized")
        except Exception as e:
            print(f"⚠️  Azure AI Manager initialization failed: {e}")
            azure_manager = None
        
        # Initialize session contexts dictionary (empty at start)
        session_contexts = {}
        print("✅ Session context storage initialized")
        
        # Initialize telemetry
        try:
            telemetry_manager = TelemetryCollector()
            print("✅ Telemetry manager initialized")
        except Exception as e:
            print(f"⚠️  Telemetry manager initialization failed: {e}")
            telemetry_manager = None
            
    except Exception as e:
        print(f"❌ Service initialization failed: {e}")

def get_or_create_session_context(session_id: str = None) -> ConversationContextManager:
    """Get existing session context or create a new one"""
    if not session_id:
        session_id = f"session_{int(time.time())}"
    
    if session_id not in session_contexts:
        session_contexts[session_id] = ConversationContextManager(
            session_id=session_id,
            max_history=15
        )
        print(f"✅ Created new session context: {session_id}")
    
    return session_contexts[session_id]

def clear_session_context(session_id: str):
    """Clear a specific session context"""
    if session_id in session_contexts:
        session_contexts[session_id].clear_conversation()
        del session_contexts[session_id]
        print(f"🧹 Cleared session context: {session_id}")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("🚀 Starting Hybrid LLM Router API...")
    initialize_services()
    print("✅ API server ready!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hybrid LLM Router API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/route", response_model=QueryResponse)
async def route_query(request: QueryRequest):
    """Route a query through the hybrid system"""
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    try:
        # Log the incoming request
        if telemetry_manager:
            telemetry_manager.log_query_received(
                query=request.query,
                session_id=request.session_id or "default",
                query_id=query_id,
                user_metadata={"strategy": request.strategy}
            )
        
        # Process with hybrid router
        if hybrid_router:
            # Use hybrid router with smart three-tier routing
            try:
                response_text = hybrid_router.route(request.query, show_reasoning=True)
                
                # Parse the response to extract source information
                source = "hybrid"
                if "[LOCAL]" in response_text:
                    source = "local"
                elif "[APIM]" in response_text:
                    source = "apim"
                elif "[FOUNDRY-AGENT]" in response_text:
                    source = "foundry"
                elif "[AZURE]" in response_text:
                    source = "cloud"
                
                # Clean response text from routing tags
                for tag in ["[LOCAL]", "[APIM]", "[FOUNDRY-AGENT]", "[AZURE]", "[LOCAL*]", "[APIM*]", "[FOUNDRY*]", "[AZURE*]"]:
                    response_text = response_text.replace(tag, "").strip()
                
                response_time = time.time() - start_time
                
                # Add to session context using ConversationContextManager
                session_context = get_or_create_session_context(request.session_id or "default")
                session_context.add_exchange(
                    user_message=request.query,
                    ai_response=response_text,
                    source=source,
                    response_time=response_time,
                    metadata={
                        'strategy': request.strategy,
                        'router_type': 'hybrid_foundry_apim',
                        'source': source,
                        'query_id': query_id
                    }
                )
                
                # Log success
                if telemetry_manager:
                    telemetry_manager.log_model_response(
                        model_type=source,
                        response_time=response_time,
                        success=True,
                        session_id=request.session_id or "default",
                        query_id=query_id,
                        response_details={
                            "router_type": "hybrid_foundry_apim",
                            "response": response_text[:100] + "..." if len(response_text) > 100 else response_text
                        }
                    )
                
                return QueryResponse(
                    response=response_text,
                    source=source,
                    responseTime=response_time,
                    metadata={
                        "strategy": request.strategy,
                        "router_type": "hybrid_foundry_apim",
                        "routing_system": "HybridFoundryAPIMRouter",
                        "timestamp": datetime.now().isoformat()
                    }
                )
            except Exception as e:
                print(f"Hybrid router error: {e}, falling back to mock response")
        
        # Fallback to mock response if router not available
        response_time = time.time() - start_time
        return QueryResponse(
            response=f"Mock response for \"{request.query[:30]}...\" using {request.strategy} strategy. This is a simulated response demonstrating the hybrid routing system.",
            source="mock",
            responseTime=response_time,
            metadata={
                "strategy": request.strategy,
                "mock": True,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    except Exception as e:
        # Log error
        if telemetry_manager:
            telemetry_manager.log_error(
                error=e,
                context="query_processing",
                session_id=request.session_id or "default",
                query_id=query_id,
                additional_data={"query_length": len(request.query)}
            )
        
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/route/bert", response_model=QueryResponse)
async def route_query_bert(request: QueryRequest):
    """Route a query specifically through BERT router"""
    start_time = time.time()
    query_id = str(uuid.uuid4())
    
    try:
        # Check if BERT router is available
        if not check_bert_router():
            return QueryResponse(
                response="BERT router is not available. Please ensure the BERT model is loaded.",
                source="error",
                responseTime=time.time() - start_time,
                metadata={
                    "strategy": "bert",
                    "error": "BERT router not available",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Log the incoming request
        if telemetry_manager:
            telemetry_manager.log_query_received(
                query=request.query,
                session_id=request.session_id or "default",
                query_id=query_id,
                user_metadata={"strategy": "bert"}
            )
        
        # Use BERT router for classification and routing
        try:
            classification = hybrid_router.bert_router.classify_query(request.query)
            
            # Route based on BERT classification
            if classification.get('route') == 'local':
                response_text = hybrid_router._query_local(request.query)
                source = "local"
            elif classification.get('route') == 'cloud':
                response_text = hybrid_router._query_azure_openai(request.query)
                source = "cloud"
            else:
                response_text = hybrid_router._query_apim(request.query)
                source = "apim"
            
            response_time = time.time() - start_time
            
            # Add to session context using ConversationContextManager
            session_context = get_or_create_session_context(request.session_id or "default")
            session_context.add_exchange(
                user_message=request.query,
                ai_response=response_text,
                source=source,
                response_time=response_time,
                metadata={
                    'strategy': 'bert',
                    'router_type': 'bert',
                    'classification': classification,
                    'query_id': query_id
                }
            )
            
            # Log success
            if telemetry_manager:
                telemetry_manager.log_model_response(
                    model_type="bert",
                    response_time=response_time,
                    success=True,
                    session_id=request.session_id or "default",
                    query_id=query_id,
                    response_details={
                        "router_type": "bert",
                        "classification": classification,
                        "routed_to": source
                    }
                )
            
            return QueryResponse(
                response=response_text,
                source=f"bert->{source}",
                responseTime=response_time,
                metadata={
                    "strategy": "bert",
                    "classification": classification,
                    "routed_to": source,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            # Fallback to regular hybrid routing
            response_text = hybrid_router.route(request.query, show_reasoning=True)
            source = "bert-fallback"
            
            response_time = time.time() - start_time
            
            return QueryResponse(
                response=response_text,
                source=source,
                responseTime=response_time,
                metadata={
                    "strategy": "bert",
                    "fallback": True,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    except Exception as e:
        # Log error
        if telemetry_manager:
            telemetry_manager.log_error(
                error=e,
                context="bert_query_processing",
                session_id=request.session_id or "default",
                query_id=query_id,
                additional_data={"query_length": len(request.query)}
            )
        
        raise HTTPException(status_code=500, detail=f"BERT query processing failed: {str(e)}")

@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status and available routers"""
    try:
        # Check local model availability
        local_available = await check_local_model()
        
        # Check cloud model availability  
        cloud_available = await check_cloud_model()
        
        # Determine system health
        if local_available and cloud_available:
            health = "healthy"
        elif local_available or cloud_available:
            health = "degraded"
        else:
            health = "error"
        
        # Check BERT and PHI router availability
        bert_available = check_bert_router()
        phi_available = check_phi_router()
        
        return SystemStatusResponse(
            availableRouters={
                "hybrid": hybrid_router is not None and (local_available or cloud_available),
                "hybrid_foundry_apim": hybrid_router is not None,
                "bert": bert_available,
                "phi": phi_available,
                "local": local_available,
                "cloud": cloud_available
            },
            systemHealth=health
        )
    
    except Exception as e:
        return SystemStatusResponse(
            availableRouters={
                "hybrid": False,
                "hybrid_foundry_apim": False,
                "bert": False,
                "phi": False,
                "local": False,
                "cloud": False
            },
            systemHealth="error"
        )

async def check_local_model() -> bool:
    """Check if local model is available"""
    try:
        import requests
        local_endpoint = os.getenv("LOCAL_MODEL_ENDPOINT", "http://localhost:60632")
        
        # Try a simple request to check if the service is up
        response = requests.get(f"{local_endpoint}/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False

async def check_cloud_model() -> bool:
    """Check if cloud model is available"""
    try:
        if azure_manager:
            # Try a simple test request
            test_response = azure_manager.chat_completion(
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        return False
    except:
        return False

def check_bert_router() -> bool:
    """Check if BERT router is available"""
    try:
        if hybrid_router and hasattr(hybrid_router, 'bert_router'):
            return hybrid_router.bert_router is not None
        return False
    except:
        return False

def check_phi_router() -> bool:
    """Check if PHI router is available"""
    try:
        if hybrid_router and hasattr(hybrid_router, 'phi_router'):
            return hybrid_router.phi_router is not None
        return False
    except:
        return False

@app.get("/session/{session_id}/context")
async def get_session_context(session_id: str):
    """Get conversation context for a session"""
    if session_id in session_contexts:
        context = session_contexts[session_id]
        return {
            "session_id": session_id,
            "chat_history": context.chat_history,
            "session_summary": context.get_session_summary(),
            "total_exchanges": len(context.chat_history)
        }
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.delete("/session/{session_id}/context")
async def clear_session_context_endpoint(session_id: str):
    """Clear conversation context for a session"""
    clear_session_context(session_id)
    return {"message": f"Session context cleared for {session_id}"}

@app.get("/session/{session_id}/insights")
async def get_session_insights(session_id: str):
    """Get conversation insights for a session"""
    if session_id in session_contexts:
        context = session_contexts[session_id]
        if context.chat_history:
            # Calculate insights similar to streamlit_multiturn_demo.py
            total_time = sum(ex['response_time'] for ex in context.chat_history)
            avg_response_time = total_time / len(context.chat_history)
            
            sources = [ex['source'] for ex in context.chat_history]
            source_counts = {source: sources.count(source) for source in set(sources)}
            
            context_used_count = sum(1 for ex in context.chat_history if ex.get('metadata', {}).get('context_used', False))
            
            return {
                "total_exchanges": len(context.chat_history),
                "avg_response_time": avg_response_time,
                "fastest_response": min(ex['response_time'] for ex in context.chat_history),
                "slowest_response": max(ex['response_time'] for ex in context.chat_history),
                "source_distribution": source_counts,
                "context_usage_rate": (context_used_count / len(context.chat_history)) * 100 if context.chat_history else 0,
                "session_summary": context.get_session_summary()
            }
        else:
            return {"message": "No conversation data available"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "hybrid_router": hybrid_router is not None,
            "azure_manager": azure_manager is not None,
            "session_contexts": len(session_contexts),
            "telemetry_manager": telemetry_manager is not None
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Hybrid LLM Router API Server...")
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )