#!/usr/bin/env python3
"""
Enhanced FastAPI Backend API for Hybrid AI Router React Demo

This backend mirrors the functionality of streamlit_multiturn_demo.py
and provides all the API endpoints needed by the React frontend.
"""

import os
import sys
import time
import uuid
import asyncio
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Add the current directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import your existing modules
try:
    from modules.hybrid_router import create_hybrid_router_from_env, HybridFoundryAPIMRouter
    from modules.context_manager import ConversationContextManager
    from modules.bert_router import BertQueryRouter, BertRouterConfig
    from modules.phi_router import PhiQueryRouter, PhiRouterConfig
    router_available = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Running in mock mode without actual routing")
    router_available = False

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str
    strategy: str = "hybrid"
    session_id: Optional[str] = None
    context_enabled: bool = True

class QueryResponse(BaseModel):
    response: str
    source: str
    response_time: float
    session_id: str
    metadata: Dict[str, Any]

class SystemStatusResponse(BaseModel):
    systemHealth: str
    availableRouters: Dict[str, bool]
    hybrid_router_targets: Optional[Dict[str, bool]] = None

class ConversationInsights(BaseModel):
    total_exchanges: int
    avg_response_time: float
    fastest_response: float
    slowest_response: float
    source_distribution: Dict[str, int]
    model_switches: int
    context_usage_rate: float
    session_duration: float

# Enhanced conversation context for session management
class EnhancedHybridRouterAPI:
    """Enhanced API class matching streamlit functionality with ConversationContextManager"""
    
    def __init__(self):
        self.session_contexts: Dict[str, ConversationContextManager] = {}
        
        # Initialize routers
        self.hybrid_router = None
        self.bert_router = None
        self.phi_router = None
        
        # Try to initialize actual routers
        if router_available:
            try:
                self.hybrid_router = create_hybrid_router_from_env()
                print("✅ Hybrid Router initialized")
            except Exception as e:
                print(f"⚠️ Hybrid Router initialization failed: {e}")
            
            try:
                bert_config = BertRouterConfig(
                    model_path="./notebooks/mobilbert_query_router_trained",
                    confidence_threshold=0.7
                )
                self.bert_router = BertQueryRouter(bert_config)
                print("✅ BERT Router initialized")
            except Exception as e:
                print(f"⚠️ BERT Router initialization failed: {e}")
            
            try:
                phi_config = PhiRouterConfig(model_path="./notebooks/phi_router_model")
                self.phi_router = PhiQueryRouter(phi_config)
                print("✅ PHI Router initialized")
            except Exception as e:
                print(f"⚠️ PHI Router initialization failed: {e}")
        else:
            print("⚠️ Running in mock mode - router modules not available")
    
    def get_or_create_session_context(self, session_id: Optional[str] = None) -> ConversationContextManager:
        """Get existing session context or create new one"""
        if not session_id:
            session_id = f"session_{int(time.time())}"
        
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = ConversationContextManager(
                session_id=session_id,
                max_history=15
            )
            print(f"✅ Created new session context: {session_id}")
        
        return self.session_contexts[session_id]
    
    async def route_query(self, query: str, strategy: str, session_id: Optional[str] = None, 
                         context_enabled: bool = True) -> QueryResponse:
        """Enhanced query routing with context management"""
        start_time = time.time()
        session_context = self.get_or_create_session_context(session_id)
        
        # Prepare context if enabled
        context = ""
        if context_enabled and session_context.chat_history:
            try:
                recent_messages = session_context.get_messages_for_model('both', include_system=False)
                context = session_context.get_conversation_context(recent_messages[-3:]) if recent_messages else ""
            except Exception as e:
                print(f"Context preparation failed: {e}")
        
        # Route the query based on strategy
        try:
            if strategy == "hybrid" and self.hybrid_router:
                response = await self._route_hybrid(query, context)
            elif strategy == "bert" and self.bert_router:
                response = await self._route_bert(query, context)
            elif strategy == "phi" and self.phi_router:
                response = await self._route_phi(query, context)
            elif strategy == "rule_based":
                response = await self._route_rule_based(query, context)
            else:
                # Fallback to mock response
                response = await self._route_mock(query, strategy)
        
        except Exception as e:
            print(f"Routing failed for {strategy}: {e}")
            response = await self._route_mock(query, strategy, error=True)
        
        response_time = time.time() - start_time
        
        # Add exchange to session context using ConversationContextManager
        session_context.add_exchange(
            user_message=query,
            ai_response=response["content"],
            source=response["source"],
            response_time=response_time,
            metadata={
                "strategy": strategy,
                "context_used": bool(context),
                "routing_decision": response.get("routing_decision", ""),
                "confidence": response.get("confidence", 1.0),
                "error": response.get("error", False)
            }
        )
        
        # Prepare metadata
        metadata = {
            "context_used": bool(context),
            "context_length": len(context) if context else 0,
            "strategy": strategy,
            "routing_decision": response.get("routing_decision", ""),
            "exchange_number": len(session_context.chat_history),
            "confidence": response.get("confidence", 1.0),
            "error": response.get("error", False)
        }
        
        return QueryResponse(
            response=response["content"],
            source=response["source"],
            response_time=response_time,
            session_id=session_context.session_id,
            metadata=metadata
        )
    
    async def _route_hybrid(self, query: str, context: str) -> Dict[str, Any]:
        """Route using hybrid router"""
        try:
            # Use hybrid router with context
            if context:
                full_query = f"{context}\n\nCurrent question: {query}"
            else:
                full_query = query
                
            response_text = self.hybrid_router.route(full_query, show_reasoning=True)
            
            # Extract source from response tags
            source = "hybrid"
            if "[LOCAL]" in response_text:
                source = "local"
            elif "[APIM]" in response_text:
                source = "apim"
            elif "[FOUNDRY-AGENT]" in response_text:
                source = "foundry"
            elif "[AZURE]" in response_text:
                source = "azure"
            
            # Clean response text
            for tag in ["[LOCAL]", "[APIM]", "[FOUNDRY-AGENT]", "[AZURE]", "[LOCAL*]", "[APIM*]", "[FOUNDRY*]", "[AZURE*]"]:
                response_text = response_text.replace(tag, "").strip()
            
            return {
                "content": response_text,
                "source": source,
                "routing_decision": "hybrid_foundry_apim_routing",
                "confidence": 0.9
            }
        except Exception as e:
            print(f"Hybrid routing failed: {e}")
            return await self._route_mock(query, "hybrid", error=True)
    
    async def _route_bert(self, query: str, context: str) -> Dict[str, Any]:
        """Route using BERT router"""
        try:
            classification = self.bert_router.classify_query(query)
            
            # Mock response based on BERT classification
            if classification.get("complexity", "medium") == "simple":
                source = "local"
                response = f"BERT classified this as simple: {query[:50]}..."
            else:
                source = "cloud"
                response = f"BERT routed to cloud for: {query[:50]}..."
            
            return {
                "content": response,
                "source": source,
                "routing_decision": f"bert_classification_{classification.get('complexity', 'unknown')}",
                "confidence": classification.get("confidence", 0.8)
            }
        except Exception as e:
            print(f"BERT routing failed: {e}")
            return await self._route_mock(query, "bert", error=True)
    
    async def _route_phi(self, query: str, context: str) -> Dict[str, Any]:
        """Route using PHI router"""
        try:
            # Simulate PHI routing logic
            routing_decision = "phi_analysis"
            
            if len(query) < 20:
                source = "local"
                response = f"PHI local response: {query}"
            else:
                source = "foundry"
                response = f"PHI foundry response for complex query: {query[:50]}..."
            
            return {
                "content": response,
                "source": source,
                "routing_decision": routing_decision,
                "confidence": 0.85
            }
        except Exception as e:
            print(f"PHI routing failed: {e}")
            return await self._route_mock(query, "phi", error=True)
    
    async def _route_rule_based(self, query: str, context: str) -> Dict[str, Any]:
        """Route using rule-based logic"""
        query_lower = query.lower()
        
        # Simple rule-based routing
        if any(word in query_lower for word in ["hello", "hi", "time", "date"]):
            source = "local"
            response = f"Rule-based local response: {query}"
        elif any(word in query_lower for word in ["complex", "analysis", "design", "architecture"]):
            source = "foundry"
            response = f"Rule-based foundry response for: {query[:50]}..."
        else:
            source = "apim"
            response = f"Rule-based APIM response: {query[:50]}..."
        
        return {
            "content": response,
            "source": source,
            "routing_decision": "rule_based_analysis",
            "confidence": 0.7
        }
    
    async def _route_mock(self, query: str, strategy: str, error: bool = False) -> Dict[str, Any]:
        """Generate mock response when actual routing fails"""
        if error:
            return {
                "content": f"Mock response due to {strategy} router error: {query[:50]}...",
                "source": "mock",
                "routing_decision": f"{strategy}_fallback",
                "confidence": 0.5,
                "error": True
            }
        
        # Generate varied mock responses
        mock_sources = ["local", "apim", "foundry", "cloud"]
        source = mock_sources[len(query) % len(mock_sources)]
        
        return {
            "content": f"Mock {strategy} response from {source}: {query[:50]}...",
            "source": source,
            "routing_decision": f"mock_{strategy}",
            "confidence": 0.6
        }
    
    def get_system_status(self) -> SystemStatusResponse:
        """Get current system status"""
        router_status = {
            "hybrid": self.hybrid_router is not None,
            "rule_based": True,  # Always available
            "bert": self.bert_router is not None,
            "phi": self.phi_router is not None
        }
        
        # Determine overall health
        if all(router_status.values()):
            health = "healthy"
        elif any(router_status.values()):
            health = "degraded"
        else:
            health = "error"
        
        # Hybrid router targets (if hybrid is available)
        hybrid_targets = None
        if self.hybrid_router:
            hybrid_targets = {
                "local": True,
                "apim": True,
                "foundry": True
            }
        
        return SystemStatusResponse(
            systemHealth=health,
            availableRouters=router_status,
            hybrid_router_targets=hybrid_targets
        )
    
    def get_conversation_insights(self, session_id: str) -> Optional[ConversationInsights]:
        """Get conversation insights for a session"""
        if session_id not in self.session_contexts:
            return None
        
        session_context = self.session_contexts[session_id]
        
        if not session_context.chat_history:
            return None
        
        response_times = [ex["response_time"] for ex in session_context.chat_history]
        sources = [ex["source"] for ex in session_context.chat_history]
        
        source_distribution = {}
        for source in sources:
            source_distribution[source] = source_distribution.get(source, 0) + 1
        
        context_used_count = sum(1 for ex in session_context.chat_history if ex.get("metadata", {}).get("context_used", False))
        
        # Get session summary for additional metrics
        session_summary = session_context.get_session_summary()
        
        return ConversationInsights(
            total_exchanges=len(session_context.chat_history),
            avg_response_time=sum(response_times) / len(response_times),
            fastest_response=min(response_times),
            slowest_response=max(response_times),
            source_distribution=source_distribution,
            model_switches=session_summary['routing_stats'].get('model_switches', 0),
            context_usage_rate=(context_used_count / len(session_context.chat_history)) * 100,
            session_duration=float(session_summary['session_info']['duration'].split(':')[0]) * 60 + float(session_summary['session_info']['duration'].split(':')[1])
        )
    
    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.session_contexts:
            self.session_contexts[session_id].clear_conversation()
            del self.session_contexts[session_id]
            print(f"🧹 Cleared session context: {session_id}")

# Initialize the enhanced API
enhanced_api = EnhancedHybridRouterAPI()

# FastAPI app setup
app = FastAPI(title="Enhanced Hybrid AI Router API", version="2.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
@app.post("/api/query", response_model=QueryResponse)
async def route_query(request: QueryRequest):
    """Route a query using the specified strategy"""
    try:
        return await enhanced_api.route_query(
            query=request.query,
            strategy=request.strategy,
            session_id=request.session_id,
            context_enabled=request.context_enabled
        )
    except Exception as e:
        print(f"Query routing error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query routing failed: {str(e)}")

@app.get("/api/system-status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get current system status"""
    try:
        return enhanced_api.get_system_status()
    except Exception as e:
        print(f"System status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

@app.get("/api/conversation-insights/{session_id}")
async def get_conversation_insights(session_id: str):
    """Get conversation insights for a session"""
    try:
        insights = enhanced_api.get_conversation_insights(session_id)
        if insights is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return insights
    except HTTPException:
        raise
    except Exception as e:
        print(f"Insights error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")

@app.delete("/api/clear-context/{session_id}")
async def clear_context(session_id: str):
    """Clear conversation context for a session"""
    try:
        enhanced_api.clear_session(session_id)
        return {"message": "Context cleared successfully"}
    except Exception as e:
        print(f"Clear context error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear context: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Enhanced Hybrid AI Router API...")
    print("📊 Features: Multi-turn conversations, context management, multiple routing strategies")
    print("🔗 Frontend: React app at http://localhost:3000")
    print("🌐 API: Available at http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )