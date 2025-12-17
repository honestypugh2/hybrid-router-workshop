"""
Hybrid Agent Context Manager Module

This module provides a unified conversation management approach that combines:
1. Agent Framework native threads for cloud/Foundry agents
2. Custom message tracking for routing analytics
3. Cross-model conversation history

Implements Microsoft's recommended pattern for thread-based persistence while
maintaining hybrid routing analytics capabilities.
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


class ConversationAnalytics:
    """Separate analytics tracking for routing decisions."""
    
    def __init__(self):
        self.exchanges = []
        self.source_counts = {}
        self.response_times = {}
    
    def record_exchange(self, source: str, response_time: float):
        """Record an exchange for analytics."""
        self.exchanges.append({'source': source, 'time': response_time})
        self.source_counts[source] = self.source_counts.get(source, 0) + 1
        
        if source not in self.response_times:
            self.response_times[source] = []
        self.response_times[source].append(response_time)
    
    def get_distribution(self) -> Dict[str, float]:
        """Get percentage distribution of sources."""
        total = len(self.exchanges)
        if total == 0:
            return {}
        return {k: (v / total * 100) for k, v in self.source_counts.items()}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics by source."""
        metrics = {}
        for source, times in self.response_times.items():
            if times:
                metrics[source] = {
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times)
                }
        return metrics
    
    def count_switches(self) -> int:
        """Count model switches in conversation."""
        switches = 0
        for i in range(1, len(self.exchanges)):
            if self.exchanges[i]['source'] != self.exchanges[i-1]['source']:
                switches += 1
        return switches
    
    def get_summary(self) -> Dict:
        """Get analytics summary."""
        return {
            'source_counts': self.source_counts,
            'total_exchanges': len(self.exchanges),
            'distribution': self.get_distribution(),
            'performance': self.get_performance_metrics(),
            'switches': self.count_switches()
        }
    
    def restore_from_dict(self, data: Dict):
        """Restore analytics from saved data."""
        self.source_counts = data.get('source_counts', {})
        # Reconstruct exchanges from source counts (approximate)
        for source, count in self.source_counts.items():
            for _ in range(count):
                self.exchanges.append({'source': source, 'time': 0.0})


class HybridAgentContextManager:
    """
    Unified conversation manager that supports:
    1. Agent Framework native threads for cloud/Foundry agents
    2. Custom message tracking for routing analytics
    3. Cross-model conversation history
    
    This implements the recommended hybrid approach combining Microsoft's
    Agent Framework best practices with custom routing analytics.
    """
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        
        # Agent Framework native thread support
        self.agent_thread = None  # Will hold AgentThread when using Agent Framework
        self.agent_instance = None  # Reference to agent for deserialization
        
        # Custom tracking for hybrid routing analytics
        self.routing_history = []  # Track routing decisions
        self.conversation_analytics = ConversationAnalytics()
        
        # Message storage in both formats
        self.messages_openai_format = []  # For non-Agent Framework models
        self.routing_metadata = []  # Enhanced routing information
        
        print(f"🔄 HybridAgentContextManager initialized")
        print(f"   Session: {self.session_id}")
        print(f"   Supports: Agent Framework threads + Custom analytics")
    
    async def initialize_agent_thread(self, agent):
        """
        Initialize native Agent Framework thread.
        
        Args:
            agent: Agent Framework agent instance
            
        Returns:
            AgentThread object
        """
        self.agent_instance = agent
        self.agent_thread = agent.get_new_thread()
        print(f"✅ Agent Framework thread initialized")
        return self.agent_thread
    
    async def add_exchange_with_agent(self, agent, prompt: str, source: str, 
                                     metadata: Dict[str, Any] = None) -> tuple:
        """
        Use Agent Framework native thread for conversation.
        Captures routing metadata alongside native thread state.
        
        Args:
            agent: Agent Framework agent instance
            prompt: User prompt
            source: Routing source (local/cloud/foundry/apim)
            metadata: Optional metadata dictionary
            
        Returns:
            (response_text, response_time)
        """
        if not self.agent_thread:
            self.agent_thread = agent.get_new_thread()
            self.agent_instance = agent
        
        start_time = time.time()
        
        # Run with native thread (maintains Agent Framework state)
        response = await agent.run(prompt, thread=self.agent_thread)
        
        response_time = time.time() - start_time
        
        # Track routing metadata separately
        routing_info = {
            'timestamp': datetime.now().isoformat(),
            'user_message': prompt,
            'response': response.text,
            'source': source,
            'response_time': response_time,
            'thread_message_count': len(self.agent_thread.messages) if hasattr(self.agent_thread, 'messages') else None,
            'metadata': metadata or {}
        }
        self.routing_metadata.append(routing_info)
        self.routing_history.append({'source': source, 'time': response_time, 'prompt': prompt})
        
        # Update analytics
        self.conversation_analytics.record_exchange(source, response_time)
        
        return response.text, response_time
    
    def add_exchange_with_local(self, prompt: str, response: str, 
                               response_time: float, metadata: Dict[str, Any] = None) -> None:
        """
        Add exchange from local model (non-Agent Framework).
        Maintains OpenAI format for compatibility.
        
        Args:
            prompt: User prompt
            response: Model response
            response_time: Response generation time
            metadata: Optional metadata dictionary
        """
        # Add to OpenAI format messages
        self.messages_openai_format.append({
            'role': 'user',
            'content': prompt
        })
        self.messages_openai_format.append({
            'role': 'assistant',
            'content': response
        })
        
        # Track routing
        routing_info = {
            'timestamp': datetime.now().isoformat(),
            'user_message': prompt,
            'response': response,
            'source': 'local',
            'response_time': response_time,
            'metadata': metadata or {}
        }
        self.routing_metadata.append(routing_info)
        self.routing_history.append({'source': 'local', 'time': response_time, 'prompt': prompt})
        
        self.conversation_analytics.record_exchange('local', response_time)
    
    def add_exchange_generic(self, prompt: str, response: str, source: str,
                            response_time: float, metadata: Dict[str, Any] = None) -> None:
        """
        Add exchange from any model source (generic method).
        
        Args:
            prompt: User prompt
            response: Model response
            source: Source identifier (local/cloud/apim/foundry)
            response_time: Response generation time
            metadata: Optional metadata dictionary
        """
        # Add to OpenAI format messages
        self.messages_openai_format.append({
            'role': 'user',
            'content': prompt
        })
        self.messages_openai_format.append({
            'role': 'assistant',
            'content': response
        })
        
        # Track routing
        routing_info = {
            'timestamp': datetime.now().isoformat(),
            'user_message': prompt,
            'response': response,
            'source': source,
            'response_time': response_time,
            'metadata': metadata or {}
        }
        self.routing_metadata.append(routing_info)
        self.routing_history.append({'source': source, 'time': response_time, 'prompt': prompt})
        
        self.conversation_analytics.record_exchange(source, response_time)
    
    def get_messages_for_local_model(self, max_messages: int = 10) -> List[Dict]:
        """
        Get messages in OpenAI format for local models.
        
        Args:
            max_messages: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        return self.messages_openai_format[-max_messages:]
    
    async def get_messages_for_agent(self) -> Optional[Any]:
        """
        Get native thread for Agent Framework.
        
        Returns:
            AgentThread object or None
        """
        return self.agent_thread
    
    async def persist_to_storage(self, storage_path: str = None) -> bool:
        """
        Persist both Agent Framework thread AND routing analytics.
        
        Args:
            storage_path: Path to save conversation data
            
        Returns:
            True if successful, False otherwise
        """
        if storage_path is None:
            storage_path = f"conversation_{self.session_id}.json"
        
        try:
            persist_data = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'routing_metadata': self.routing_metadata,
                'routing_history': self.routing_history,
                'analytics': self.conversation_analytics.get_summary(),
                'openai_messages': self.messages_openai_format
            }
            
            # Serialize Agent Framework thread if available
            if self.agent_thread:
                persist_data['agent_thread'] = await self.agent_thread.serialize()
                persist_data['has_agent_thread'] = True
            else:
                persist_data['has_agent_thread'] = False
            
            # Save to storage
            with open(storage_path, 'w', encoding='utf-8') as f:
                json.dump(persist_data, f, indent=2, default=str)
            
            print(f"💾 Conversation persisted to: {storage_path}")
            return True
            
        except Exception as e:
            print(f"❌ Persistence failed: {e}")
            return False
    
    async def resume_from_storage(self, storage_path: str, agent=None) -> bool:
        """
        Resume conversation from storage.
        Restores both Agent Framework thread and routing analytics.
        
        Args:
            storage_path: Path to saved conversation data
            agent: Agent instance (required if thread was saved)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Restore routing metadata and analytics
            self.session_id = data['session_id']
            self.routing_metadata = data.get('routing_metadata', [])
            self.routing_history = data.get('routing_history', [])
            self.messages_openai_format = data.get('openai_messages', [])
            
            # Restore analytics
            if 'analytics' in data:
                self.conversation_analytics.restore_from_dict(data['analytics'])
            
            # Restore Agent Framework thread if available
            if data.get('has_agent_thread') and 'agent_thread' in data and agent:
                self.agent_instance = agent
                self.agent_thread = await agent.deserialize_thread(data['agent_thread'])
                print(f"✅ Agent Framework thread restored")
            
            print(f"✅ Conversation resumed from: {storage_path}")
            print(f"   - Exchanges: {len(self.routing_metadata)}")
            print(f"   - Has thread: {data.get('has_agent_thread', False)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Resume failed: {e}")
            return False
    
    def get_routing_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive routing and performance analytics.
        
        Returns:
            Dictionary with routing statistics and metrics
        """
        analytics_summary = self.conversation_analytics.get_summary()
        
        return {
            'session_id': self.session_id,
            'total_exchanges': len(self.routing_metadata),
            'routing_distribution': analytics_summary['distribution'],
            'performance_metrics': analytics_summary['performance'],
            'model_switches': analytics_summary['switches'],
            'recent_routing': self.routing_metadata[-5:] if len(self.routing_metadata) > 0 else [],
            'has_agent_thread': self.agent_thread is not None,
            'message_formats': {
                'openai_format_count': len(self.messages_openai_format),
                'agent_thread_active': self.agent_thread is not None
            }
        }
    
    def get_conversation_flow(self) -> List[Dict[str, Any]]:
        """
        Get conversation flow showing routing decisions.
        
        Returns:
            List of exchanges with routing information
        """
        flow = []
        for i, metadata in enumerate(self.routing_metadata, 1):
            flow.append({
                'exchange': i,
                'timestamp': metadata['timestamp'],
                'source': metadata['source'],
                'response_time': metadata['response_time'],
                'user_message_preview': metadata['user_message'][:50] + '...' if len(metadata['user_message']) > 50 else metadata['user_message']
            })
        return flow
    
    def clear_conversation(self):
        """Clear all conversation data and reset analytics."""
        self.agent_thread = None
        self.agent_instance = None
        self.routing_history.clear()
        self.conversation_analytics = ConversationAnalytics()
        self.messages_openai_format.clear()
        self.routing_metadata.clear()
        
        print(f"🧹 Conversation cleared for session {self.session_id}")
    
    def get_recent_exchanges(self, count: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent conversation exchanges (router compatibility method).
        
        Args:
            count: Number of recent exchanges to return
            
        Returns:
            List of exchange dictionaries with full message content
        """
        return self.routing_metadata[-count:] if len(self.routing_metadata) > 0 else []
    
    def get_conversation_messages(self, count: int = 10) -> List[Dict[str, str]]:
        """
        Get recent conversation in simple message format for display.
        
        Args:
            count: Number of recent message pairs to return
            
        Returns:
            List of dictionaries with 'user' and 'assistant' keys
        """
        exchanges = self.get_recent_exchanges(count)
        messages = []
        for exchange in exchanges:
            messages.append({
                'user': exchange.get('user_message', ''),
                'assistant': exchange.get('response', ''),
                'source': exchange.get('source', 'unknown'),
                'timestamp': exchange.get('timestamp', ''),
                'response_time': exchange.get('response_time', 0.0)
            })
        return messages
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get session summary (router compatibility method).
        
        Returns:
            Dictionary with session information
        """
        analytics_summary = self.conversation_analytics.get_summary()
        
        return {
            'session_info': {
                'session_id': self.session_id,
                'total_exchanges': len(self.routing_metadata),
                'has_agent_thread': self.agent_thread is not None,
                'started': self.routing_metadata[0]['timestamp'] if self.routing_metadata else None,
                'last_activity': self.routing_metadata[-1]['timestamp'] if self.routing_metadata else None
            },
            'conversation_flow': self.get_conversation_flow(),
            'routing_distribution': analytics_summary['distribution'],
            'performance_metrics': analytics_summary['performance']
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get conversation summary statistics (router compatibility method).
        
        Returns:
            Dictionary with conversation statistics
        """
        return {
            'total_messages': len(self.messages_openai_format),
            'total_exchanges': len(self.routing_metadata),
            'routing_history_count': len(self.routing_history),
            'session_id': self.session_id
        }
    
    def export_conversation(self, filename: str = None) -> str:
        """
        Export conversation to JSON file (router compatibility method).
        
        Args:
            filename: Optional filename for export
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"conversation_{self.session_id}_{int(time.time())}.json"
        
        export_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'exchanges': self.routing_metadata,
            'routing_history': self.routing_history,
            'analytics': self.conversation_analytics.get_summary(),
            'messages': self.messages_openai_format
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"💾 Conversation exported to: {filename}")
        return filename



# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_hybrid_manager():
        """Test the HybridAgentContextManager."""
        print("🧪 Testing HybridAgentContextManager")
        print("=" * 60)
        
        # Create manager
        manager = HybridAgentContextManager("test_session_001")
        
        # Test adding local model exchanges
        print("\n📝 Adding local model exchanges...")
        manager.add_exchange_with_local(
            prompt="Hello, how are you?",
            response="I'm doing well, thank you!",
            response_time=0.12,
            metadata={"type": "greeting"}
        )
        
        manager.add_exchange_generic(
            prompt="What's 2+2?",
            response="2+2 equals 4.",
            source="local",
            response_time=0.08,
            metadata={"type": "math"}
        )
        
        # Simulate APIM exchange
        manager.add_exchange_generic(
            prompt="Analyze enterprise architecture patterns",
            response="Enterprise architecture patterns include microservices, event-driven, and layered architectures...",
            source="apim",
            response_time=1.45,
            metadata={"type": "enterprise", "complex": True}
        )
        
        # Get routing summary
        print("\n📊 Routing Summary:")
        summary = manager.get_routing_summary()
        print(f"   Total exchanges: {summary['total_exchanges']}")
        print(f"   Model switches: {summary['model_switches']}")
        print(f"   Has agent thread: {summary['has_agent_thread']}")
        
        print(f"\n🎯 Distribution:")
        for source, percentage in summary['routing_distribution'].items():
            print(f"   {source}: {percentage:.1f}%")
        
        # Test persistence
        print("\n💾 Testing persistence...")
        await manager.persist_to_storage("test_conversation.json")
        
        # Test resume
        print("\n🔄 Testing resume...")
        new_manager = HybridAgentContextManager("resumed_session")
        await new_manager.resume_from_storage("test_conversation.json")
        
        # Cleanup
        if os.path.exists("test_conversation.json"):
            os.remove("test_conversation.json")
            print("🧹 Test file removed")
        
        print("\n✅ HybridAgentContextManager test completed!")
    
    # Run test
    asyncio.run(test_hybrid_manager())
