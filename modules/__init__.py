"""
Hybrid LLM Router Workshop Modules

This package contains the core modules for the hybrid LLM system:
- router: Intelligent query routing between local and cloud models
- context_manager: Conversation history and context management
- telemetry: Observability and performance monitoring
"""

from .router import HybridRouter, ModelTarget, QueryAnalysis, analyze_query_characteristics, route_query
from .context_manager import ConversationManager, ConversationMessage, MessageRole, ModelSource
from .telemetry import TelemetryCollector, EventType, MetricType, TelemetryEvent, PerformanceMetric

__version__ = "1.0.0"

__all__ = [
    # Router components
    "HybridRouter",
    "ModelTarget", 
    "QueryAnalysis",
    "analyze_query_characteristics",
    "route_query",
    
    # Context management components
    "ConversationManager",
    "ConversationMessage", 
    "MessageRole",
    "ModelSource",
    
    # Telemetry components
    "TelemetryCollector",
    "EventType",
    "MetricType", 
    "TelemetryEvent",
    "PerformanceMetric"
]