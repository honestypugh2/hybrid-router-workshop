"""
Telemetry and Observability Module

This module provides comprehensive logging, monitoring, and analytics
capabilities for the hybrid LLM system.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager

# Optional Azure Monitor integration
try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    AZURE_MONITOR_AVAILABLE = True
except ImportError:
    AZURE_MONITOR_AVAILABLE = False
    trace = None
    metrics = None


class EventType(Enum):
    """Types of telemetry events"""
    QUERY_RECEIVED = "query_received"
    ROUTING_DECISION = "routing_decision"
    MODEL_REQUEST = "model_request"
    MODEL_RESPONSE = "model_response"
    ERROR = "error"
    CONVERSATION_START = "conversation_start"
    CONVERSATION_END = "conversation_end"
    PERFORMANCE_METRIC = "performance_metric"


class MetricType(Enum):
    """Types of performance metrics"""
    RESPONSE_TIME = "response_time"
    TOKEN_COUNT = "token_count"
    MODEL_SWITCH = "model_switch"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class TelemetryEvent:
    """Represents a telemetry event"""
    event_type: EventType
    timestamp: str
    session_id: str
    query_id: str
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class PerformanceMetric:
    """Represents a performance metric"""
    metric_type: MetricType
    value: float
    unit: str
    timestamp: str
    tags: Dict[str, str]


class TelemetryCollector:
    """
    Collects and manages telemetry data for the hybrid LLM system.
    
    Features:
    - Local file logging
    - Console output
    - Azure Monitor integration (optional)
    - Performance metrics collection
    - Error tracking
    """

    def __init__(self, 
                 enable_console_logging: bool = True,
                 enable_file_logging: bool = True,
                 log_file_path: Optional[str] = None,
                 enable_azure_monitor: bool = False,
                 azure_connection_string: Optional[str] = None):
        """
        Initialize telemetry collector.
        
        Args:
            enable_console_logging: Enable console output
            enable_file_logging: Enable file logging
            log_file_path: Path for log file (auto-generated if None)
            enable_azure_monitor: Enable Azure Monitor integration
            azure_connection_string: Azure Monitor connection string
        """
        self.enable_console_logging = enable_console_logging
        self.enable_file_logging = enable_file_logging
        self.enable_azure_monitor = enable_azure_monitor and AZURE_MONITOR_AVAILABLE
        
        # Setup logging
        self._setup_logger(log_file_path)
        
        # Setup Azure Monitor if requested
        if self.enable_azure_monitor and azure_connection_string:
            self._setup_azure_monitor(azure_connection_string)
        
        # Initialize metrics storage
        self.session_metrics: Dict[str, List[PerformanceMetric]] = {}
        self.session_events: Dict[str, List[TelemetryEvent]] = {}
        
        # Performance counters
        self.counters = {
            'total_queries': 0,
            'local_responses': 0,
            'cloud_responses': 0,
            'errors': 0,
            'model_switches': 0
        }
        
        self.start_time = datetime.now(timezone.utc)

    def _setup_logger(self, log_file_path: Optional[str]):
        """Setup logging configuration"""
        self.logger = logging.getLogger('hybrid_llm_telemetry')
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.enable_file_logging:
            if log_file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file_path = f"hybrid_llm_telemetry_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.log_file_path = log_file_path

    def _setup_azure_monitor(self, connection_string: str):
        """Setup Azure Monitor integration"""
        try:
            configure_azure_monitor(connection_string=connection_string)
            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)
            
            # Create custom metrics
            self.response_time_histogram = self.meter.create_histogram(
                name="response_time",
                description="Response time in seconds",
                unit="s"
            )
            
            self.query_counter = self.meter.create_counter(
                name="queries_total",
                description="Total number of queries"
            )
            
            self.logger.info("Azure Monitor telemetry configured successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to configure Azure Monitor: {e}")
            self.enable_azure_monitor = False

    def start_query_span(self, query: str, session_id: str, query_id: str) -> Optional[Any]:
        """Start a distributed trace span for a query"""
        if not self.enable_azure_monitor or not self.tracer:
            return None
        
        try:
            span = self.tracer.start_span(
                name="hybrid_llm_query",
                attributes={
                    "session_id": session_id,
                    "query_id": query_id,
                    "query_length": len(query.split()),
                    "query_preview": query[:50]
                }
            )
            return span
        except Exception as e:
            self.logger.error(f"Failed to start query span: {e}")
            return None

    def log_event(self, event_type: EventType, session_id: str, query_id: str, 
                  data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """Log a telemetry event"""
        event = TelemetryEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            query_id=query_id,
            data=data,
            metadata=metadata or {}
        )
        
        # Store event
        if session_id not in self.session_events:
            self.session_events[session_id] = []
        self.session_events[session_id].append(event)
        
        # Log to configured outputs
        log_message = self._format_event_message(event)
        self.logger.info(log_message)
        
        # Send to Azure Monitor if enabled
        if self.enable_azure_monitor and self.tracer:
            try:
                with self.tracer.start_as_current_span(f"event_{event_type.value}") as span:
                    span.set_attributes({
                        "event_type": event_type.value,
                        "session_id": session_id,
                        "query_id": query_id,
                        **{f"data_{k}": str(v) for k, v in data.items()}
                    })
            except Exception as e:
                self.logger.error(f"Failed to send event to Azure Monitor: {e}")

    def log_query_received(self, query: str, session_id: str, query_id: str, 
                          user_metadata: Optional[Dict[str, Any]] = None):
        """Log when a query is received"""
        self.counters['total_queries'] += 1
        
        data = {
            "query": query,
            "query_length": len(query),
            "word_count": len(query.split()),
            "character_count": len(query)
        }
        
        self.log_event(EventType.QUERY_RECEIVED, session_id, query_id, data, user_metadata)

    def log_routing_decision(self, query: str, target_model: str, reasoning: str,
                           complexity_score: float, session_id: str, query_id: str):
        """Log routing decision details"""
        data = {
            "target_model": target_model,
            "reasoning": reasoning,
            "complexity_score": complexity_score,
            "query_preview": query[:100]
        }
        
        self.log_event(EventType.ROUTING_DECISION, session_id, query_id, data)

    def log_model_request(self, model_type: str, session_id: str, query_id: str, 
                         request_details: Dict[str, Any]):
        """Log model request details"""
        data = {
            "model_type": model_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **request_details
        }
        
        self.log_event(EventType.MODEL_REQUEST, session_id, query_id, data)

    def log_model_response(self, model_type: str, response_time: float, 
                          success: bool, session_id: str, query_id: str,
                          response_details: Optional[Dict[str, Any]] = None):
        """Log model response details"""
        # Update counters
        if success:
            if model_type == "local":
                self.counters['local_responses'] += 1
            elif model_type == "cloud":
                self.counters['cloud_responses'] += 1
        else:
            self.counters['errors'] += 1
        
        data = {
            "model_type": model_type,
            "response_time": response_time,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if response_details:
            data.update(response_details)
        
        self.log_event(EventType.MODEL_RESPONSE, session_id, query_id, data)
        
        # Record performance metric
        self.record_metric(
            MetricType.RESPONSE_TIME,
            response_time,
            "seconds",
            session_id,
            {"model_type": model_type, "success": str(success)}
        )
        
        # Send to Azure Monitor
        if self.enable_azure_monitor and self.response_time_histogram:
            try:
                self.response_time_histogram.record(
                    response_time,
                    {"model_type": model_type, "success": str(success)}
                )
            except Exception as e:
                self.logger.error(f"Failed to record metric to Azure Monitor: {e}")

    def log_error(self, error: Exception, context: str, session_id: str, 
                  query_id: str, additional_data: Optional[Dict[str, Any]] = None):
        """Log error events"""
        self.counters['errors'] += 1
        
        data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if additional_data:
            data.update(additional_data)
        
        self.log_event(EventType.ERROR, session_id, query_id, data)
        
        # Log as error level
        self.logger.error(f"Error in {context}: {error}", exc_info=True)

    def log_model_switch(self, from_model: str, to_model: str, session_id: str, query_id: str):
        """Log when the system switches between models"""
        self.counters['model_switches'] += 1
        
        data = {
            "from_model": from_model,
            "to_model": to_model,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        self.record_metric(
            MetricType.MODEL_SWITCH,
            1.0,
            "count",
            session_id,
            {"from_model": from_model, "to_model": to_model}
        )

    def record_metric(self, metric_type: MetricType, value: float, unit: str,
                     session_id: str, tags: Optional[Dict[str, str]] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            value=value,
            unit=unit,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tags=tags or {}
        )
        
        if session_id not in self.session_metrics:
            self.session_metrics[session_id] = []
        self.session_metrics[session_id].append(metric)

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive summary for a session"""
        events = self.session_events.get(session_id, [])
        metrics = self.session_metrics.get(session_id, [])
        
        # Analyze events
        query_events = [e for e in events if e.event_type == EventType.QUERY_RECEIVED]
        response_events = [e for e in events if e.event_type == EventType.MODEL_RESPONSE]
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        
        # Calculate response time statistics
        response_times = [e.data.get('response_time', 0) for e in response_events 
                         if e.data.get('success', False)]
        
        local_times = [e.data.get('response_time', 0) for e in response_events 
                      if e.data.get('model_type') == 'local' and e.data.get('success', False)]
        
        cloud_times = [e.data.get('response_time', 0) for e in response_events 
                      if e.data.get('model_type') == 'cloud' and e.data.get('success', False)]
        
        summary = {
            'session_id': session_id,
            'total_queries': len(query_events),
            'total_responses': len(response_events),
            'successful_responses': len([e for e in response_events if e.data.get('success', False)]),
            'error_count': len(error_events),
            'local_responses': len([e for e in response_events if e.data.get('model_type') == 'local']),
            'cloud_responses': len([e for e in response_events if e.data.get('model_type') == 'cloud']),
            'model_switches': len([m for m in metrics if m.metric_type == MetricType.MODEL_SWITCH])
        }
        
        # Response time statistics
        if response_times:
            summary.update({
                'avg_response_time': sum(response_times) / len(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times)
            })
        
        if local_times:
            summary['avg_local_response_time'] = sum(local_times) / len(local_times)
        
        if cloud_times:
            summary['avg_cloud_response_time'] = sum(cloud_times) / len(cloud_times)
        
        # Calculate efficiency metrics
        if local_times and cloud_times:
            avg_local = sum(local_times) / len(local_times)
            avg_cloud = sum(cloud_times) / len(cloud_times)
            summary['speed_advantage'] = avg_cloud / avg_local if avg_local > 0 else 0
        
        return summary

    def get_global_summary(self) -> Dict[str, Any]:
        """Get global telemetry summary across all sessions"""
        runtime = datetime.now(timezone.utc) - self.start_time
        
        summary = {
            'runtime_seconds': runtime.total_seconds(),
            'runtime_minutes': runtime.total_seconds() / 60,
            'start_time': self.start_time.isoformat(),
            'counters': self.counters.copy(),
            'total_sessions': len(self.session_events),
            'azure_monitor_enabled': self.enable_azure_monitor
        }
        
        # Calculate percentages
        total_responses = self.counters['local_responses'] + self.counters['cloud_responses']
        if total_responses > 0:
            summary['local_percentage'] = (self.counters['local_responses'] / total_responses) * 100
            summary['cloud_percentage'] = (self.counters['cloud_responses'] / total_responses) * 100
            summary['error_rate'] = (self.counters['errors'] / self.counters['total_queries']) * 100
        
        return summary

    def export_telemetry(self, filename: Optional[str] = None) -> str:
        """Export all telemetry data to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"telemetry_export_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'global_summary': self.get_global_summary(),
            'session_events': {sid: [e.to_dict() for e in events] 
                             for sid, events in self.session_events.items()},
            'session_metrics': {sid: [asdict(m) for m in metrics] 
                              for sid, metrics in self.session_metrics.items()},
            'session_summaries': {sid: self.get_session_summary(sid) 
                                for sid in self.session_events.keys()}
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Telemetry data exported to {filename}")
        return filename

    def _format_event_message(self, event: TelemetryEvent) -> str:
        """Format event for logging output"""
        data_str = json.dumps(event.data, default=str, separators=(',', ':'))
        return f"[{event.event_type.value.upper()}] Session:{event.session_id} Query:{event.query_id} Data:{data_str}"

    @contextmanager
    def trace_operation(self, operation_name: str, session_id: str, 
                       query_id: str, **attributes):
        """Context manager for tracing operations"""
        start_time = time.time()
        span = None
        
        if self.enable_azure_monitor and self.tracer:
            try:
                span = self.tracer.start_span(
                    name=operation_name,
                    attributes={
                        "session_id": session_id,
                        "query_id": query_id,
                        **attributes
                    }
                )
            except Exception as e:
                self.logger.error(f"Failed to start trace span: {e}")
        
        try:
            yield span
            if span:
                span.set_status(Status(StatusCode.OK))
        except Exception as e:
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            duration = time.time() - start_time
            self.record_metric(
                MetricType.PERFORMANCE_METRIC,
                duration,
                "seconds",
                session_id,
                {"operation": operation_name}
            )
            
            if span:
                try:
                    span.end()
                except Exception as e:
                    self.logger.error(f"Failed to end trace span: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Initialize telemetry collector
    telemetry = TelemetryCollector(
        enable_console_logging=True,
        enable_file_logging=True,
        enable_azure_monitor=False  # Set to True with connection string for Azure
    )
    
    print("📊 Telemetry System Test")
    print("=" * 30)
    
    # Simulate a conversation session
    session_id = "test_session_001"
    
    # Query 1
    query_id = "query_001"
    telemetry.log_query_received("Hello, how are you?", session_id, query_id)
    telemetry.log_routing_decision("Hello, how are you?", "local", "Simple greeting", 0.1, session_id, query_id)
    telemetry.log_model_request("local", session_id, query_id, {"max_tokens": 100})
    telemetry.log_model_response("local", 0.15, True, session_id, query_id, {"tokens_generated": 25})
    
    # Query 2
    query_id = "query_002"
    telemetry.log_query_received("Analyze the quarterly business performance", session_id, query_id)
    telemetry.log_routing_decision("Analyze the quarterly business performance", "cloud", "Complex analysis", 0.8, session_id, query_id)
    telemetry.log_model_switch("local", "cloud", session_id, query_id)
    telemetry.log_model_request("cloud", session_id, query_id, {"max_tokens": 500})
    telemetry.log_model_response("cloud", 2.45, True, session_id, query_id, {"tokens_generated": 150})
    
    # Show session summary
    print(f"\n📈 Session Summary:")
    summary = telemetry.get_session_summary(session_id)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Show global summary
    print(f"\n🌍 Global Summary:")
    global_summary = telemetry.get_global_summary()
    for key, value in global_summary.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        elif isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Export telemetry
    export_file = telemetry.export_telemetry("test_telemetry.json")
    print(f"\n💾 Telemetry exported to: {export_file}")
    
    print("\n✅ Telemetry system test completed!")