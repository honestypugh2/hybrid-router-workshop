"""
Azure AI Manager Module

This module provides a unified interface for managing and interacting with Azure AI services,
including Azure AI Foundry Agents and direct Azure OpenAI API calls. It offers intelligent
routing, conversation management, and comprehensive error handling for enterprise applications.

Classes:
    AzureAIConfig: Configuration dataclass for Azure AI services
    AzureAIManager: Main manager class for Azure AI operations
    
Functions:
    create_azure_manager_from_env: Create manager from environment variables
    
Example:
    ```python
    from modules.azure_ai_manager import AzureAIManager, AzureAIConfig
    
    # Create manager with configuration
    config = AzureAIConfig(
        foundry_endpoint="your-foundry-endpoint",
        azure_endpoint="your-azure-endpoint",
        azure_key="your-api-key"
    )
    manager = AzureAIManager(config)
    
    # Query with automatic method selection
    response, time, success = manager.query("Hello, how are you?")
    ```
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
from openai import AzureOpenAI

# Optional Azure AI Foundry dependencies
try:
    from azure.ai.projects import AIProjectClient
    from azure.ai.agents.models import MessageRole, RunStatus
    from azure.identity import DefaultAzureCredential
    FOUNDRY_AVAILABLE = True
except ImportError:
    FOUNDRY_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AzureAIConfig:
    """
    Configuration for Azure AI services.
    
    Attributes:
        foundry_endpoint (Optional[str]): Azure AI Foundry project endpoint
        azure_endpoint (str): Azure OpenAI endpoint URL
        azure_key (str): Azure OpenAI API key
        azure_deployment (str): Azure OpenAI deployment name
        azure_api_version (str): Azure OpenAI API version
        agent_name (str): Name for Foundry agent
        agent_instructions (str): Instructions for Foundry agent
        max_tokens (int): Maximum tokens for responses
        temperature (float): Temperature for response generation
        timeout (int): Request timeout in seconds
    """
    foundry_endpoint: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_key: Optional[str] = None
    azure_deployment: Optional[str] = None
    azure_api_version: str = "2024-02-01"
    agent_name: str = "Azure-AI-Assistant"
    agent_instructions: str = """You are an intelligent AI assistant providing helpful, accurate, and comprehensive responses."""
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 30


class AzureAIManager:
    """
    Unified manager for Azure AI services (both Foundry Agents and direct OpenAI).
    
    This class provides a high-level interface for interacting with Azure AI services,
    automatically handling service availability, error recovery, and intelligent routing
    between different Azure AI endpoints.
    
    Attributes:
        config (AzureAIConfig): Configuration for Azure AI services
        project_client (Optional[AIProjectClient]): Azure AI Foundry project client
        azure_client (Optional[AzureOpenAI]): Direct Azure OpenAI client
        foundry_agent (Optional): Created Foundry agent instance
        agent_thread (Optional): Active conversation thread
        query_history (List[Dict]): History of queries and responses
        
    Example:
        ```python
        config = AzureAIConfig(
            azure_endpoint="https://your-resource.openai.azure.com/",
            azure_key="your-api-key",
            azure_deployment="gpt-35-turbo"
        )
        
        manager = AzureAIManager(config)
        response, duration, success = manager.query("Hello, world!")
        ```
    """
    
    def __init__(self, config: Optional[AzureAIConfig] = None):
        """
        Initialize the Azure AI Manager.
        
        Args:
            config (Optional[AzureAIConfig]): Configuration for Azure AI services.
                If None, will attempt to create from environment variables.
                
        Raises:
            ValueError: If no valid configuration is provided and environment
                variables are incomplete.
        """
        self.config = config or self._create_config_from_env()
        self.project_client = None
        self.azure_client = None
        self.foundry_agent = None
        self.agent_thread = None
        self.query_history = []
        
        # Initialize clients
        self._init_azure_client()
        self._init_foundry_client()
        
        # Service availability flags
        self.use_foundry_agents = self.project_client is not None
        self.use_direct_openai = self.azure_client is not None
        
        if not self.use_foundry_agents and not self.use_direct_openai:
            raise ValueError("No Azure AI services available. Check configuration.")
        
        logger.info(f"Azure AI Manager initialized: Foundry={self.use_foundry_agents}, Direct={self.use_direct_openai}")
    
    def _create_config_from_env(self) -> AzureAIConfig:
        """Create configuration from environment variables."""
        return AzureAIConfig(
            foundry_endpoint=os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            azure_key=os.environ.get("AZURE_OPENAI_KEY"),
            azure_deployment=os.environ.get("AZURE_DEPLOYMENT_NAME"),
            azure_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        )
    
    def _init_azure_client(self):
        """Initialize direct Azure OpenAI client."""
        if self.config.azure_endpoint and self.config.azure_key:
            try:
                self.azure_client = AzureOpenAI(
                    api_key=self.config.azure_key,
                    api_version=self.config.azure_api_version,
                    azure_endpoint=self.config.azure_endpoint
                )
                logger.info("✅ Azure OpenAI client initialized")
            except Exception as e:
                logger.error(f"❌ Azure OpenAI client initialization failed: {e}")
    
    def _init_foundry_client(self):
        """Initialize Azure AI Foundry client and agent."""
        if not FOUNDRY_AVAILABLE or not self.config.foundry_endpoint:
            return
            
        try:
            credential = DefaultAzureCredential()
            self.project_client = AIProjectClient(
                endpoint=self.config.foundry_endpoint,
                credential=credential
            )
            
            # Create agent if Azure deployment is available
            if self.config.azure_deployment:
                self.foundry_agent = self.project_client.agents.create_agent(
                    model=self.config.azure_deployment,
                    name=self.config.agent_name,
                    instructions=self.config.agent_instructions,
                    description="Managed agent for Azure AI operations"
                )
                
                # Create conversation thread
                self.agent_thread = self.project_client.agents.threads.create()
                logger.info("✅ Azure AI Foundry agent and thread created")
            
            logger.info("✅ Azure AI Foundry client initialized")
            
        except Exception as e:
            logger.error(f"❌ Azure AI Foundry initialization failed: {e}")
    
    def query(self, prompt: str, method: str = "auto", **kwargs) -> Tuple[str, float, bool]:
        """
        Query Azure AI with automatic or specified method selection.
        
        Args:
            prompt (str): The input prompt/query
            method (str): Method to use - 'auto', 'foundry', 'direct'
            **kwargs: Additional parameters for the query
            
        Returns:
            Tuple[str, float, bool]: (response_content, response_time, success)
            
        Example:
            ```python
            response, time, success = manager.query("What is AI?")
            response, time, success = manager.query("Complex analysis", method="foundry")
            ```
        """
        start_time = time.time()
        
        try:
            if method == "auto":
                return self._query_intelligent(prompt, **kwargs)
            elif method == "foundry" and self.use_foundry_agents:
                return self._query_foundry_agent(prompt, **kwargs)
            elif method == "direct" and self.use_direct_openai:
                return self._query_direct_openai(prompt, **kwargs)
            else:
                error_msg = f"Method '{method}' not available or not configured"
                logger.warning(error_msg)
                return error_msg, time.time() - start_time, False
                
        except Exception as e:
            error_msg = f"Query failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, time.time() - start_time, False
    
    def _query_intelligent(self, prompt: str, **kwargs) -> Tuple[str, float, bool]:
        """
        Intelligent query routing with automatic fallback.
        
        Tries Foundry Agents first (if available), then falls back to direct OpenAI.
        """
        if self.use_foundry_agents:
            response, response_time, success = self._query_foundry_agent(prompt, **kwargs)
            if success:
                return response, response_time, success
            logger.warning("Foundry agent failed, falling back to direct OpenAI")
        
        if self.use_direct_openai:
            return self._query_direct_openai(prompt, **kwargs)
        
        return "No Azure AI services available", 0, False
    
    def _query_foundry_agent(self, prompt: str, **kwargs) -> Tuple[str, float, bool]:
        """
        Query using Azure AI Foundry Agent.
        
        Args:
            prompt (str): The input prompt
            **kwargs: Additional parameters
            
        Returns:
            Tuple[str, float, bool]: (response, time, success)
        """
        if not self.foundry_agent or not self.agent_thread:
            return "Foundry Agent not available", 0, False
        
        try:
            start_time = time.time()
            
            # Add message to thread
            message = self.project_client.agents.messages.create(
                thread_id=self.agent_thread.id,
                role=MessageRole.USER,
                content=prompt
            )
            
            # Create and execute run
            run = self.project_client.agents.runs.create_and_process(
                thread_id=self.agent_thread.id,
                agent_id=self.foundry_agent.id
            )
            
            # Wait for completion with timeout
            max_wait = kwargs.get('timeout', self.config.timeout)
            elapsed = 0
            while run.status in [RunStatus.IN_PROGRESS, RunStatus.QUEUED] and elapsed < max_wait:
                time.sleep(0.5)
                elapsed += 0.5
                run = self.project_client.agents.runs.get(
                    thread_id=self.agent_thread.id, 
                    run_id=run.id
                )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if run.status == RunStatus.COMPLETED:
                # Get latest response
                messages = self.project_client.agents.messages.list(thread_id=self.agent_thread.id)

                # Convert ItemPaged to list and get the most recent message
                message_list = list(messages)
                if message_list:
                    latest_message = message_list[0]  # Most recent message
                
                if latest_message.role == MessageRole.ASSISTANT:
                    # content = latest_message.content[0].text.value
                    # Handle different content types
                    if hasattr(latest_message.content[0], 'text'):
                        content = latest_message.content[0].text.value
                    else:
                        content = str(latest_message.content[0])
                    self._record_query(prompt, content, response_time, "foundry", True)
                    return content, response_time, True
            
            error_msg = f"Agent run failed or timed out: {run.status}"
            self._record_query(prompt, error_msg, response_time, "foundry", False)
            return error_msg, response_time, False
            
        except Exception as e:
            error_msg = f"Foundry Agent error: {str(e)}"
            response_time = time.time() - start_time
            self._record_query(prompt, error_msg, response_time, "foundry", False)
            return error_msg, response_time, False
    
    def _query_direct_openai(self, prompt: str, **kwargs) -> Tuple[str, float, bool]:
        """
        Query using direct Azure OpenAI API.
        
        Args:
            prompt (str): The input prompt
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
            
        Returns:
            Tuple[str, float, bool]: (response, time, success)
        """
        if not self.azure_client:
            return "Direct Azure OpenAI not available", 0, False
        
        try:
            start_time = time.time()
            
            # Prepare parameters
            max_tokens = kwargs.get('max_tokens', self.config.max_tokens)
            temperature = kwargs.get('temperature', self.config.temperature)
            
            response = self.azure_client.chat.completions.create(
                model=self.config.azure_deployment,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            content = response.choices[0].message.content
            self._record_query(prompt, content, response_time, "direct", True)
            return content, response_time, True
            
        except Exception as e:
            error_msg = f"Direct OpenAI error: {str(e)}"
            response_time = time.time() - start_time
            self._record_query(prompt, error_msg, response_time, "direct", False)
            return error_msg, response_time, False
    
    def _record_query(self, prompt: str, response: str, response_time: float, method: str, success: bool):
        """Record query in history for analytics."""
        self.query_history.append({
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'response': response,
            'response_time': response_time,
            'method': method,
            'success': success
        })
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get information about available Azure AI capabilities.
        
        Returns:
            Dict[str, Any]: Comprehensive capabilities information
        """
        return {
            "foundry_agents_available": self.use_foundry_agents,
            "direct_openai_available": self.use_direct_openai,
            "primary_method": "foundry" if self.use_foundry_agents else "direct" if self.use_direct_openai else "none",
            "deployment": self.config.azure_deployment,
            "endpoint": self.config.azure_endpoint,
            "agent_name": self.config.agent_name if self.foundry_agent else None,
            "agent_id": self.foundry_agent.id if self.foundry_agent else None,
            "thread_id": self.agent_thread.id if self.agent_thread else None,
            "total_queries": len(self.query_history),
            "successful_queries": sum(1 for q in self.query_history if q['success'])
        }
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """
        Get detailed query statistics.
        
        Returns:
            Dict[str, Any]: Query statistics and performance metrics
        """
        if not self.query_history:
            return {"message": "No queries recorded"}
        
        successful_queries = [q for q in self.query_history if q['success']]
        foundry_queries = [q for q in self.query_history if q['method'] == 'foundry']
        direct_queries = [q for q in self.query_history if q['method'] == 'direct']
        
        successful_times = [q['response_time'] for q in successful_queries]
        avg_time = sum(successful_times) / len(successful_times) if successful_times else 0
        
        return {
            "total_queries": len(self.query_history),
            "successful_queries": len(successful_queries),
            "success_rate": len(successful_queries) / len(self.query_history) * 100,
            "foundry_queries": len(foundry_queries),
            "direct_queries": len(direct_queries),
            "average_response_time": avg_time,
            "fastest_response": min(successful_times) if successful_times else 0,
            "slowest_response": max(successful_times) if successful_times else 0
        }
    
    def create_conversation_thread(self) -> Optional[Any]:
        """
        Create a new conversation thread (Foundry Agents only).
        
        Returns:
            Optional[Any]: New thread object or None if not available
        """
        if self.use_foundry_agents and self.project_client:
            try:
                new_thread = self.project_client.agents.create_thread()
                logger.info(f"New conversation thread created: {new_thread.id}")
                return new_thread
            except Exception as e:
                logger.error(f"Failed to create thread: {e}")
                return None
        else:
            logger.warning("Foundry Agents not available for thread creation")
            return None
    
    def switch_thread(self, thread_id: str) -> bool:
        """
        Switch to a different conversation thread.
        
        Args:
            thread_id (str): ID of the thread to switch to
            
        Returns:
            bool: True if switch was successful, False otherwise
        """
        if not self.use_foundry_agents:
            logger.warning("Thread switching requires Foundry Agents")
            return False
        
        try:
            # Validate thread exists by attempting to list its messages
            messages = self.project_client.agents.messages.list(thread_id=thread_id)
            
            # Create a simple thread object for compatibility
            class ThreadProxy:
                def __init__(self, thread_id):
                    self.id = thread_id
            
            self.agent_thread = ThreadProxy(thread_id)
            logger.info(f"Switched to thread: {thread_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to thread {thread_id}: {e}")
            return False
    
    def reset(self):
        """Reset the manager by clearing history and creating new thread."""
        self.query_history.clear()
        if self.use_foundry_agents:
            self.agent_thread = self.create_conversation_thread()
        logger.info("Azure AI Manager reset completed")


def create_azure_manager_from_env() -> AzureAIManager:
    """
    Create Azure AI Manager using environment variables.
    
    Required environment variables:
    - AZURE_AI_FOUNDRY_ENDPOINT: Azure AI Foundry project endpoint
    - AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL
    - AZURE_OPENAI_KEY: Azure OpenAI API key  
    - AZURE_DEPLOYMENT_NAME: Azure OpenAI deployment name
    
    Optional environment variables:
    - AZURE_OPENAI_API_VERSION: API version (default: "2024-02-01")
    
    Returns:
        AzureAIManager: Configured manager instance
        
    Raises:
        ValueError: If required environment variables are missing
    """
    required_vars = ["AZURE_AI_FOUNDRY_ENDPOINT", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_KEY", "AZURE_DEPLOYMENT_NAME"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    return AzureAIManager()