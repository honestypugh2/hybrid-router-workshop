"""
Conversation Context Management Module

This module manages conversation history and context sharing across
local and cloud models in the hybrid LLM system.
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class MessageRole(Enum):
    """Message roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ModelSource(Enum):
    """Model sources for responses"""
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"
    BERT = "bert"
    PHI = "phi"
    APIM = "apim"
    FOUNDRY = "foundry"
    ERROR = "error"
    SYSTEM = "system"


@dataclass
class ConversationMessage:
    """Represents a single message in the conversation"""
    role: MessageRole
    content: str
    timestamp: str
    source: Optional[ModelSource] = None
    response_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_openai_format(self) -> Dict[str, str]:
        """Convert to OpenAI API message format"""
        return {
            "role": self.role.value,
            "content": self.content
        }


class ConversationManager:
    """
    Manages conversation history and context for hybrid chat sessions.
    
    This class handles:
    - Message storage and retrieval
    - Context window management for different model types
    - Conversation statistics and analytics
    - Export/import functionality
    """

    def __init__(self, max_history_length: int = 20, max_tokens_per_model: int = 2000):
        """
        Initialize conversation manager.
        
        Args:
            max_history_length: Maximum number of message pairs to retain
            max_tokens_per_model: Rough token limit for context window management
        """
        self.max_history_length = max_history_length
        self.max_tokens_per_model = max_tokens_per_model
        self.conversation_history: List[ConversationMessage] = []
        
        self.conversation_stats = {
            'total_exchanges': 0,
            'local_responses': 0,
            'cloud_responses': 0,
            'model_switches': 0,
            'start_time': datetime.now(),
            'last_model_used': None,
            'total_tokens_estimated': 0,
            'average_response_time': 0.0,
            'errors_count': 0
        }

    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConversationMessage:
        """
        Add a user message to the conversation history.
        
        Args:
            content: Message content
            metadata: Optional metadata dictionary
            
        Returns:
            Created ConversationMessage object
        """
        message = ConversationMessage(
            role=MessageRole.USER,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        self.conversation_history.append(message)
        self._update_token_count(content)
        
        return message

    def add_assistant_message(self, content: str, source: ModelSource, 
                            response_time: float, metadata: Optional[Dict[str, Any]] = None) -> ConversationMessage:
        """
        Add an assistant message to the conversation history.
        
        Args:
            content: Response content
            source: Model source (local, cloud, etc.)
            response_time: Time taken to generate response
            metadata: Optional metadata dictionary
            
        Returns:
            Created ConversationMessage object
        """
        # Clean source tags from content for storage
        clean_content = content
        if content.startswith('[LOCAL]'):
            clean_content = content[7:].strip()
            source = ModelSource.LOCAL
        elif content.startswith('[CLOUD]'):
            clean_content = content[7:].strip()
            source = ModelSource.CLOUD
        elif content.startswith('[ERROR]'):
            clean_content = content[7:].strip()
            source = ModelSource.ERROR

        message = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content=clean_content,
            timestamp=datetime.now().isoformat(),
            source=source,
            response_time=response_time,
            metadata=metadata or {}
        )
        
        self.conversation_history.append(message)
        self._update_statistics(source, response_time)
        self._update_token_count(clean_content)
        
        return message

    def add_system_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> ConversationMessage:
        """
        Add a system message to the conversation history.
        
        Args:
            content: System message content
            metadata: Optional metadata dictionary
            
        Returns:
            Created ConversationMessage object
        """
        message = ConversationMessage(
            role=MessageRole.SYSTEM,
            content=content,
            timestamp=datetime.now().isoformat(),
            source=ModelSource.SYSTEM,
            metadata=metadata or {}
        )
        
        self.conversation_history.append(message)
        return message

    def add_message(self, session_id: str, message: Dict[str, str]) -> None:
        """
        Add a message to the conversation history in OpenAI format.
        
        Args:
            session_id: Session identifier (currently unused but kept for API compatibility)
            message: Message dictionary with 'role' and 'content' keys
        """
        role = message.get('role', 'user')
        content = message.get('content', '')
        
        if role == 'user':
            self.add_user_message(content, metadata={'session_id': session_id})
        elif role == 'assistant':
            # Default to hybrid source since we don't have detailed source info
            self.add_assistant_message(
                content=content, 
                source=ModelSource.HYBRID,
                response_time=0.0,  # Default response time
                metadata={'session_id': session_id}
            )
        elif role == 'system':
            self.add_system_message(content, metadata={'session_id': session_id})

    def get_messages_for_model(self, target_model: str = 'both', 
                              include_system: bool = True) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for model input.
        
        Args:
            target_model: 'local', 'cloud', or 'both' - determines context length
            include_system: Whether to include system messages
            
        Returns:
            List of message dictionaries suitable for OpenAI API
        """
        # Filter messages based on include_system parameter
        messages = []
        for msg in self.conversation_history:
            if not include_system and msg.role == MessageRole.SYSTEM:
                continue
            messages.append(msg.to_openai_format())
        
        # Apply length limits based on target model
        if target_model == 'local':
            # Local models have smaller context windows
            max_messages = min(10, self.max_history_length)
        elif target_model == 'cloud':
            # Cloud models can handle more context
            max_messages = self.max_history_length
        else:
            # Default to moderate length
            max_messages = 15
        
        # Truncate if necessary (keep most recent messages)
        if len(messages) > max_messages:
            # Always keep system messages at the beginning if they exist
            system_messages = [msg for msg in messages if msg['role'] == 'system']
            conversation_messages = [msg for msg in messages if msg['role'] != 'system']
            
            # Take the most recent conversation messages
            recent_messages = conversation_messages[-(max_messages - len(system_messages)):]
            messages = system_messages + recent_messages
        
        return messages

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the conversation statistics."""
        duration = datetime.now() - self.conversation_stats['start_time']
        total_responses = (self.conversation_stats['local_responses'] + 
                          self.conversation_stats['cloud_responses'])
        
        if total_responses > 0:
            local_percentage = (self.conversation_stats['local_responses'] / total_responses) * 100
            cloud_percentage = (self.conversation_stats['cloud_responses'] / total_responses) * 100
        else:
            local_percentage = cloud_percentage = 0
        
        return {
            'total_exchanges': self.conversation_stats['total_exchanges'],
            'local_responses': self.conversation_stats['local_responses'],
            'cloud_responses': self.conversation_stats['cloud_responses'],
            'local_percentage': local_percentage,
            'cloud_percentage': cloud_percentage,
            'model_switches': self.conversation_stats['model_switches'],
            'duration_minutes': duration.total_seconds() / 60,
            'messages_in_history': len(self.conversation_history),
            'estimated_total_tokens': self.conversation_stats['total_tokens_estimated'],
            'average_response_time': self.conversation_stats['average_response_time'],
            'errors_count': self.conversation_stats['errors_count'],
            'efficiency_metrics': self._calculate_efficiency_metrics()
        }

    def _calculate_efficiency_metrics(self) -> Dict[str, float]:
        """Calculate efficiency metrics for the conversation."""
        local_messages = [msg for msg in self.conversation_history 
                         if msg.source == ModelSource.LOCAL and msg.response_time is not None]
        cloud_messages = [msg for msg in self.conversation_history 
                         if msg.source == ModelSource.CLOUD and msg.response_time is not None]
        
        metrics = {}
        
        if local_messages:
            local_times = [msg.response_time for msg in local_messages]
            metrics['local_avg_time'] = sum(local_times) / len(local_times)
            metrics['local_min_time'] = min(local_times)
            metrics['local_max_time'] = max(local_times)
        
        if cloud_messages:
            cloud_times = [msg.response_time for msg in cloud_messages]
            metrics['cloud_avg_time'] = sum(cloud_times) / len(cloud_times)
            metrics['cloud_min_time'] = min(cloud_times)
            metrics['cloud_max_time'] = max(cloud_times)
        
        # Calculate potential time savings
        if local_messages and cloud_messages:
            avg_local = metrics['local_avg_time']
            avg_cloud = metrics['cloud_avg_time']
            total_local_responses = len(local_messages)
            
            # Estimate time if all queries went to cloud
            estimated_all_cloud_time = avg_cloud * (len(local_messages) + len(cloud_messages))
            actual_total_time = sum([msg.response_time for msg in local_messages + cloud_messages])
            
            metrics['time_saved_seconds'] = estimated_all_cloud_time - actual_total_time
            if estimated_all_cloud_time > 0:
                metrics['efficiency_percentage'] = (metrics['time_saved_seconds'] / estimated_all_cloud_time) * 100
        
        return metrics

    def export_conversation(self, filename: Optional[str] = None, 
                          include_metadata: bool = True) -> str:
        """
        Export conversation history to JSON file.
        
        Args:
            filename: Output filename (auto-generated if None)
            include_metadata: Whether to include message metadata
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_export_{timestamp}.json"
        
        # Prepare export data
        messages_data = []
        for msg in self.conversation_history:
            msg_dict = {
                'role': msg.role.value,
                'content': msg.content,
                'timestamp': msg.timestamp,
                'source': msg.source.value if msg.source else None,
                'response_time': msg.response_time
            }
            
            if include_metadata and msg.metadata:
                msg_dict['metadata'] = msg.metadata
            
            messages_data.append(msg_dict)
        
        export_data = {
            'conversation_history': messages_data,
            'statistics': self.get_conversation_summary(),
            'export_timestamp': datetime.now().isoformat(),
            'total_messages': len(self.conversation_history),
            'configuration': {
                'max_history_length': self.max_history_length,
                'max_tokens_per_model': self.max_tokens_per_model
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
        
        return filename

    def import_conversation(self, filename: str) -> bool:
        """
        Import conversation history from JSON file.
        
        Args:
            filename: Path to JSON file to import
            
        Returns:
            True if import successful, False otherwise
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Clear current conversation
            self.clear_history()
            
            # Import messages
            for msg_data in data.get('conversation_history', []):
                role = MessageRole(msg_data['role'])
                content = msg_data['content']
                timestamp = msg_data['timestamp']
                source = ModelSource(msg_data['source']) if msg_data.get('source') else None
                response_time = msg_data.get('response_time')
                metadata = msg_data.get('metadata', {})
                
                message = ConversationMessage(
                    role=role,
                    content=content,
                    timestamp=timestamp,
                    source=source,
                    response_time=response_time,
                    metadata=metadata
                )
                
                self.conversation_history.append(message)
            
            # Restore configuration if available
            config = data.get('configuration', {})
            if 'max_history_length' in config:
                self.max_history_length = config['max_history_length']
            if 'max_tokens_per_model' in config:
                self.max_tokens_per_model = config['max_tokens_per_model']
            
            # Recalculate statistics
            self._recalculate_statistics()
            
            return True
        
        except Exception as e:
            print(f"Error importing conversation: {e}")
            return False

    def clear_history(self):
        """Clear conversation history and reset statistics."""
        self.conversation_history = []
        self.conversation_stats = {
            'total_exchanges': 0,
            'local_responses': 0,
            'cloud_responses': 0,
            'model_switches': 0,
            'start_time': datetime.now(),
            'last_model_used': None,
            'total_tokens_estimated': 0,
            'average_response_time': 0.0,
            'errors_count': 0
        }

    def get_last_n_messages(self, n: int, include_system: bool = False) -> List[ConversationMessage]:
        """
        Get the last N messages from the conversation.
        
        Args:
            n: Number of messages to retrieve
            include_system: Whether to include system messages
            
        Returns:
            List of ConversationMessage objects
        """
        messages = self.conversation_history
        if not include_system:
            messages = [msg for msg in messages if msg.role != MessageRole.SYSTEM]
        
        return messages[-n:] if n > 0 else messages

    def search_messages(self, query: str, case_sensitive: bool = False) -> List[ConversationMessage]:
        """
        Search for messages containing specific text.
        
        Args:
            query: Search query
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of matching ConversationMessage objects
        """
        if not case_sensitive:
            query = query.lower()
        
        matches = []
        for msg in self.conversation_history:
            content = msg.content if case_sensitive else msg.content.lower()
            if query in content:
                matches.append(msg)
        
        return matches

    def _update_statistics(self, source: ModelSource, response_time: float):
        """Update conversation statistics."""
        self.conversation_stats['total_exchanges'] += 1
        
        if source == ModelSource.LOCAL:
            self.conversation_stats['local_responses'] += 1
        elif source == ModelSource.CLOUD:
            self.conversation_stats['cloud_responses'] += 1
        elif source == ModelSource.ERROR:
            self.conversation_stats['errors_count'] += 1
        
        # Track model switches
        if (self.conversation_stats['last_model_used'] is not None and 
            self.conversation_stats['last_model_used'] != source):
            self.conversation_stats['model_switches'] += 1
        
        self.conversation_stats['last_model_used'] = source
        
        # Update average response time
        total_responses = (self.conversation_stats['local_responses'] + 
                          self.conversation_stats['cloud_responses'])
        if total_responses > 0:
            current_avg = self.conversation_stats['average_response_time']
            self.conversation_stats['average_response_time'] = (
                (current_avg * (total_responses - 1) + response_time) / total_responses
            )

    def _update_token_count(self, content: str):
        """Update estimated token count."""
        # Rough estimation: 1 token ≈ 0.75 words
        estimated_tokens = int(len(content.split()) * 1.33)
        self.conversation_stats['total_tokens_estimated'] += estimated_tokens

    def _recalculate_statistics(self):
        """Recalculate statistics from conversation history (used after import)."""
        self.conversation_stats = {
            'total_exchanges': 0,
            'local_responses': 0,
            'cloud_responses': 0,
            'model_switches': 0,
            'start_time': datetime.now(),
            'last_model_used': None,
            'total_tokens_estimated': 0,
            'average_response_time': 0.0,
            'errors_count': 0
        }
        
        # Find earliest message for start time
        if self.conversation_history:
            timestamps = [datetime.fromisoformat(msg.timestamp) for msg in self.conversation_history]
            self.conversation_stats['start_time'] = min(timestamps)
        
        # Recalculate statistics
        response_times = []
        for msg in self.conversation_history:
            if msg.role == MessageRole.ASSISTANT and msg.source:
                if msg.source == ModelSource.LOCAL:
                    self.conversation_stats['local_responses'] += 1
                elif msg.source == ModelSource.CLOUD:
                    self.conversation_stats['cloud_responses'] += 1
                elif msg.source == ModelSource.ERROR:
                    self.conversation_stats['errors_count'] += 1
                
                if msg.response_time is not None:
                    response_times.append(msg.response_time)
            
            self._update_token_count(msg.content)
        
        self.conversation_stats['total_exchanges'] = (
            self.conversation_stats['local_responses'] + 
            self.conversation_stats['cloud_responses']
        )
        
        if response_times:
            self.conversation_stats['average_response_time'] = sum(response_times) / len(response_times)
        
        # Calculate model switches
        sources = [msg.source for msg in self.conversation_history 
                  if msg.role == MessageRole.ASSISTANT and msg.source in [ModelSource.LOCAL, ModelSource.CLOUD]]
        
        switches = 0
        for i in range(1, len(sources)):
            if sources[i] != sources[i-1]:
                switches += 1
        
        self.conversation_stats['model_switches'] = switches
        if sources:
            self.conversation_stats['last_model_used'] = sources[-1]


class ConversationContextManager(ConversationManager):
    """
    Enhanced conversation context manager that aligns with HybridConversationManager 
    from lab5_hybrid_orchestration notebook.
    
    This class provides:
    - Session-based conversation management
    - Context-aware message formatting  
    - Hybrid routing compatibility
    - Exchange tracking similar to lab5
    """
    
    def __init__(self, session_id: str = None, max_history: int = 15):
        super().__init__(max_history_length=max_history)
        self.session_id = session_id or f"session_{int(time.time())}"
        self.chat_history = []  # Lab5-style chat history
        
        # Enhanced stats to match lab5
        self.conversation_stats.update({
            'apim_responses': 0,
            'foundry_responses': 0,
            'azure_responses': 0,
            'mock_responses': 0,
            'fallback_uses': 0,
            'session_start': datetime.now(),
        })
        
        print(f"🗣️ ConversationContextManager initialized for session: {self.session_id}")
    
    def add_exchange(self, user_message: str, ai_response: str, source: str, 
                    response_time: float, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Add a complete exchange (user + AI response) to conversation history.
        This method aligns with the lab5 HybridConversationManager pattern.
        
        Args:
            user_message: User's input message
            ai_response: AI's response
            source: Source of the response (local, apim, foundry, etc.)
            response_time: Time taken to generate response
            metadata: Additional metadata
            
        Returns:
            Exchange dictionary with all details
        """
        # Create exchange in lab5 format
        exchange = {
            'timestamp': datetime.now(),
            'user_message': user_message,
            'response': ai_response,
            'source': source,
            'response_time': response_time,
            'was_fallback': metadata.get('error', False) if metadata else False,
            'exchange_number': len(self.chat_history) + 1,
            'metadata': metadata or {}
        }
        
        # Add to lab5-style chat history
        self.chat_history.append(exchange)
        
        # Also add to the detailed conversation_history for compatibility
        # Add user message
        user_msg = ConversationMessage(
            role=MessageRole.USER,
            content=user_message,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.conversation_history.append(user_msg)
        
        # Add assistant message
        try:
            source_enum = ModelSource(source.lower())
        except ValueError:
            # Handle custom sources like 'apim', 'foundry'
            source_enum = ModelSource.HYBRID
        
        ai_msg = ConversationMessage(
            role=MessageRole.ASSISTANT,
            content=ai_response,
            timestamp=datetime.now().isoformat(),
            source=source_enum,
            response_time=response_time,
            metadata=metadata or {}
        )
        self.conversation_history.append(ai_msg)
        
        # Update statistics
        self._update_enhanced_stats(source, response_time, exchange['was_fallback'])
        self._update_token_count(user_message)
        self._update_token_count(ai_response)
        
        return exchange
    
    def _update_enhanced_stats(self, source: str, response_time: float, was_fallback: bool):
        """Update enhanced statistics compatible with lab5 format."""
        self.conversation_stats['total_exchanges'] += 1
        
        # Track by source type (enhanced for lab5 compatibility)
        source_lower = source.lower()
        if 'local' in source_lower:
            self.conversation_stats['local_responses'] += 1
        elif 'apim' in source_lower:
            self.conversation_stats['apim_responses'] += 1
        elif 'foundry' in source_lower:
            self.conversation_stats['foundry_responses'] += 1
        elif 'azure' in source_lower:
            self.conversation_stats['azure_responses'] += 1
        elif 'cloud' in source_lower:
            self.conversation_stats['cloud_responses'] += 1
        elif 'mock' in source_lower:
            self.conversation_stats['mock_responses'] += 1
        
        # Track model switches
        if self.conversation_stats['last_model_used'] and self.conversation_stats['last_model_used'] != source:
            self.conversation_stats['model_switches'] += 1
        
        # Track fallback usage
        if was_fallback:
            self.conversation_stats['fallback_uses'] += 1
        
        self.conversation_stats['last_model_used'] = source
        
        # Update average response time
        total_responses = (self.conversation_stats['local_responses'] + 
                          self.conversation_stats['cloud_responses'] +
                          self.conversation_stats['apim_responses'] +
                          self.conversation_stats['foundry_responses'] +
                          self.conversation_stats['azure_responses'] +
                          self.conversation_stats['mock_responses'])
        
        if total_responses > 0:
            current_avg = self.conversation_stats['average_response_time']
            self.conversation_stats['average_response_time'] = (
                (current_avg * (total_responses - 1) + response_time) / total_responses
            )
    
    def get_conversation_context(self, recent_messages: List[Dict], max_context_length: int = 2000) -> str:
        """
        Get conversation context formatted for model input.
        This method provides context similar to lab5 format.
        
        Args:
            recent_messages: Recent conversation messages
            max_context_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        if not recent_messages:
            return ""
        
        # Format recent exchanges for context
        context_parts = []
        context_parts.append("Previous conversation context:")
        
        for msg in recent_messages[-5:]:  # Last 5 exchanges
            if msg.get('role') == 'user':
                context_parts.append(f"User: {msg['content']}")
            elif msg.get('role') == 'assistant':
                context_parts.append(f"Assistant: {msg['content']}")
        
        context = "\n".join(context_parts)
        
        # Truncate if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "...[truncated]"
        
        return context
    
    def get_recent_exchanges(self, count: int = 3) -> List[Dict[str, Any]]:
        """Get recent exchanges in lab5 format."""
        return self.chat_history[-count:] if count > 0 else self.chat_history
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary compatible with lab5 format."""
        if not self.chat_history:
            return {"message": "No conversation history available"}
        
        # Calculate session duration
        session_duration = datetime.now() - self.conversation_stats['session_start']
        
        # Analyze routing patterns
        total = self.conversation_stats['total_exchanges']
        routing_distribution = {
            'local': (self.conversation_stats['local_responses'] / total * 100) if total > 0 else 0,
            'apim': (self.conversation_stats['apim_responses'] / total * 100) if total > 0 else 0,
            'foundry': (self.conversation_stats['foundry_responses'] / total * 100) if total > 0 else 0,
            'azure': (self.conversation_stats['azure_responses'] / total * 100) if total > 0 else 0,
            'cloud': (self.conversation_stats['cloud_responses'] / total * 100) if total > 0 else 0,
            'mock': (self.conversation_stats['mock_responses'] / total * 100) if total > 0 else 0
        }
        
        return {
            'session_info': {
                'session_id': self.session_id,
                'duration': str(session_duration).split('.')[0],
                'total_exchanges': total,
                'conversation_length': len(self.chat_history)
            },
            'routing_stats': {
                'distribution': routing_distribution,
                'model_switches': self.conversation_stats['model_switches'],
                'fallback_uses': self.conversation_stats['fallback_uses'],
                'current_source': self.conversation_stats['last_model_used']
            },
            'conversation_flow': [
                {
                    'exchange': ex['exchange_number'],
                    'source': ex['source'],
                    'fallback': ex['was_fallback'],
                    'timestamp': ex['timestamp'].strftime('%H:%M:%S')
                }
                for ex in self.chat_history[-5:]  # Last 5 exchanges
            ]
        }
    
    def clear_conversation(self):
        """Clear conversation history and reset statistics."""
        self.chat_history.clear()
        self.conversation_history.clear()
        self.conversation_stats = {
            'total_exchanges': 0,
            'local_responses': 0,
            'cloud_responses': 0,
            'apim_responses': 0,
            'foundry_responses': 0,
            'azure_responses': 0,
            'mock_responses': 0,
            'model_switches': 0,
            'fallback_uses': 0,
            'start_time': datetime.now(),
            'session_start': datetime.now(),
            'last_model_used': None,
            'total_tokens_estimated': 0,
            'average_response_time': 0.0,
            'errors_count': 0
        }
        print(f"🧹 Conversation cleared for session {self.session_id}")
    
    def export_conversation(self, filename: str = None) -> str:
        """Export conversation to JSON file in lab5 format."""
        if not filename:
            filename = f"conversation_{self.session_id}.json"
        
        export_data = {
            'session_info': {
                'session_id': self.session_id,
                'start_time': self.conversation_stats['session_start'].isoformat(),
                'export_time': datetime.now().isoformat(),
                'total_exchanges': len(self.chat_history)
            },
            'conversation': [
                {
                    'exchange_number': ex['exchange_number'],
                    'timestamp': ex['timestamp'].isoformat(),
                    'user_message': ex['user_message'],
                    'response': ex['response'],
                    'source': ex['source'],
                    'response_time': ex['response_time'],
                    'was_fallback': ex['was_fallback'],
                    'metadata': ex['metadata']
                }
                for ex in self.chat_history
            ],
            'statistics': {
                key: value.isoformat() if isinstance(value, datetime) else value
                for key, value in self.conversation_stats.items()
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filename


# Example usage and testing
if __name__ == "__main__":
    # Create conversation manager
    conv_mgr = ConversationManager()
    
    print("🗣️  Conversation Manager Test")
    print("=" * 40)
    
    # Add test conversation
    conv_mgr.add_user_message("Hello! How are you?")
    conv_mgr.add_assistant_message("Hello! I'm doing well, thank you for asking.", 
                                  ModelSource.LOCAL, 0.15)
    
    conv_mgr.add_user_message("Can you analyze the stock market trends for this quarter?")
    conv_mgr.add_assistant_message("I can provide a comprehensive analysis of this quarter's stock market trends...", 
                                  ModelSource.CLOUD, 2.34)
    
    conv_mgr.add_user_message("Thanks, that was helpful!")
    conv_mgr.add_assistant_message("You're welcome! Happy to help.", 
                                  ModelSource.LOCAL, 0.08)
    
    # Show summary
    summary = conv_mgr.get_conversation_summary()
    print(f"📊 Conversation Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        elif isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value:.3f}" if isinstance(sub_value, float) else f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    # Test ConversationContextManager
    print("\n" + "=" * 40)
    print("🗣️  ConversationContextManager Test")
    
    context_mgr = ConversationContextManager("test_session")
    
    # Test adding exchanges
    context_mgr.add_exchange(
        "Hello!", 
        "Hi there! How can I help you?", 
        "local", 
        0.12,
        {"context_used": False}
    )
    
    context_mgr.add_exchange(
        "What's the weather like?", 
        "I don't have access to real-time weather data.", 
        "local", 
        0.08,
        {"context_used": True}
    )
    
    context_mgr.add_exchange(
        "Can you analyze complex data patterns?", 
        "I can help analyze complex data patterns using various techniques.", 
        "foundry", 
        2.45,
        {"context_used": True}
    )
    
    # Show session summary
    session_summary = context_mgr.get_session_summary()
    print(f"\n📊 Session Summary:")
    print(f"Session ID: {session_summary['session_info']['session_id']}")
    print(f"Duration: {session_summary['session_info']['duration']}")
    print(f"Total Exchanges: {session_summary['session_info']['total_exchanges']}")
    print(f"Model Switches: {session_summary['routing_stats']['model_switches']}")
    
    print(f"\n🎯 Routing Distribution:")
    for source, percentage in session_summary['routing_stats']['distribution'].items():
        if percentage > 0:
            print(f"   {source}: {percentage:.1f}%")
    
    # Test export
    export_file = conv_mgr.export_conversation("test_conversation.json")
    print(f"\n💾 Conversation exported to: {export_file}")
    
    print("\n✅ Context manager test completed!")