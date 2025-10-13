"""
Configuration module for Hybrid LLM Router Workshop.

This module provides comprehensive configuration management for all labs,
including Azure AI Foundry, Azure OpenAI, local models, telemetry, and routing settings.
"""

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional
import os

# Load environment variables
load_dotenv(find_dotenv(".env"))


class AzureFoundryConfig(BaseSettings):
    """Azure AI Foundry configuration settings."""
    
    azure_ai_foundry_endpoint: Optional[str] = None
    azure_ai_foundry_api_key: Optional[str] = None
    azure_ai_foundry_project_id: Optional[str] = None
    azure_ai_foundry_project_name: Optional[str] = None
    azure_ai_foundry_project_description: Optional[str] = None
    azure_ai_foundry_project_version: str = "1.0.0"
    azure_ai_foundry_project_client: Optional[str] = None
    ai_project_client: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = False


class AzureOpenAIConfig(BaseSettings):
    """Azure OpenAI configuration settings."""
    
    # Azure AI OpenAI (Foundry integration)
    azure_ai_openai_endpoint: Optional[str] = None
    azure_ai_openai_api_key: Optional[str] = None
    azure_ai_openai_project_id: Optional[str] = None
    azure_ai_openai_project_name: Optional[str] = None
    azure_ai_openai_project_description: Optional[str] = None
    azure_ai_openai_project_version: str = "1.0.0"
    
    # Direct Azure OpenAI
    azure_openai_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-02-15-preview"
    azure_deployment_name: str = "gpt-4"

    class Config:
        env_file = ".env"
        case_sensitive = False


class LocalModelConfig(BaseSettings):
    """Local model configuration settings for Foundry Local."""
    
    local_model_endpoint: str = "http://localhost:8080"
    local_model_name: str = "phi-3.5-mini"
    local_api_key: str = "not-needed"

    class Config:
        env_file = ".env"
        case_sensitive = False


class TelemetryConfig(BaseSettings):
    """Telemetry and monitoring configuration settings."""
    
    azure_monitor_connection_string: Optional[str] = None
    enable_telemetry: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False


class ApplicationConfig(BaseSettings):
    """General application configuration settings."""
    
    max_context_length: int = 4000
    response_timeout: int = 30
    debug_mode: bool = False

    class Config:
        env_file = ".env"
        case_sensitive = False


class RoutingConfig(BaseSettings):
    """Routing configuration settings for hybrid routing logic."""
    
    complexity_threshold: float = 0.5
    local_model_max_tokens: int = 500
    cloud_model_max_tokens: int = 1500

    class Config:
        env_file = ".env"
        case_sensitive = False


class AzureConfig(BaseSettings):
    """Legacy Azure configuration for backwards compatibility."""
    
    subscription_id: Optional[str] = None
    resource_group: Optional[str] = None
    workspace_name: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = False


class Config(BaseSettings):
    """
    Comprehensive configuration class that aggregates all configuration sections.
    
    This class provides easy access to all configuration settings needed across
    all labs in the Hybrid LLM Router Workshop.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize all configuration sections
        self.azure_foundry = AzureFoundryConfig()
        self.azure_openai = AzureOpenAIConfig()
        self.local_model = LocalModelConfig()
        self.telemetry = TelemetryConfig()
        self.application = ApplicationConfig()
        self.routing = RoutingConfig()
        self.azure = AzureConfig()
    
    def get_azure_foundry_endpoint(self) -> Optional[str]:
        """Get Azure AI Foundry endpoint."""
        return self.azure_foundry.azure_ai_foundry_endpoint
    
    def get_azure_openai_endpoint(self) -> Optional[str]:
        """Get Azure OpenAI endpoint (direct or via Foundry)."""
        return (self.azure_openai.azure_openai_endpoint or 
                self.azure_openai.azure_ai_openai_endpoint)
    
    def get_azure_openai_key(self) -> Optional[str]:
        """Get Azure OpenAI API key (direct or via Foundry)."""
        return (self.azure_openai.azure_openai_key or 
                self.azure_openai.azure_ai_openai_api_key)
    
    def get_local_model_endpoint(self) -> str:
        """Get local model endpoint."""
        return self.local_model.local_model_endpoint
    
    def is_telemetry_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self.telemetry.enable_telemetry and bool(self.telemetry.azure_monitor_connection_string)
    
    def get_complexity_threshold(self) -> float:
        """Get complexity threshold for routing decisions."""
        return self.routing.complexity_threshold
    
    def is_debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return self.application.debug_mode
    
    def get_environment_info(self) -> dict:
        """Get comprehensive environment information for debugging."""
        return {
            "azure_foundry": {
                "endpoint_configured": bool(self.azure_foundry.azure_ai_foundry_endpoint),
                "api_key_configured": bool(self.azure_foundry.azure_ai_foundry_api_key),
                "project_id": self.azure_foundry.azure_ai_foundry_project_id,
                "project_name": self.azure_foundry.azure_ai_foundry_project_name,
            },
            "azure_openai": {
                "direct_endpoint_configured": bool(self.azure_openai.azure_openai_endpoint),
                "direct_key_configured": bool(self.azure_openai.azure_openai_key),
                "foundry_endpoint_configured": bool(self.azure_openai.azure_ai_openai_endpoint),
                "foundry_key_configured": bool(self.azure_openai.azure_ai_openai_api_key),
                "deployment_name": self.azure_openai.azure_deployment_name,
                "api_version": self.azure_openai.azure_openai_api_version,
            },
            "local_model": {
                "endpoint": self.local_model.local_model_endpoint,
                "model_name": self.local_model.local_model_name,
            },
            "telemetry": {
                "enabled": self.is_telemetry_enabled(),
                "connection_string_configured": bool(self.telemetry.azure_monitor_connection_string),
            },
            "routing": {
                "complexity_threshold": self.routing.complexity_threshold,
                "local_max_tokens": self.routing.local_model_max_tokens,
                "cloud_max_tokens": self.routing.cloud_model_max_tokens,
            },
            "application": {
                "debug_mode": self.application.debug_mode,
                "max_context_length": self.application.max_context_length,
                "response_timeout": self.application.response_timeout,
            }
        }

    class Config:
        env_file = ".env"
        case_sensitive = False


# Create a global configuration instance
config = Config()
