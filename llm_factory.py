"""
Universal LLM Factory for supporting multiple LLM providers
Supports: Azure OpenAI, OpenAI, Claude (Anthropic), Google Gemini
"""

import os
import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
import httpx
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"
    CLAUDE = "claude"
    GEMINI = "gemini"

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    provider: LLMProvider
    model_name: Optional[str] = None
    temperature: float = 0.1
    max_retries: int = 3
    timeout: int = 300
    
    # Azure OpenAI specific
    azure_endpoint: Optional[str] = None
    azure_deployment: Optional[str] = None
    api_version: Optional[str] = None
    azure_ad_token: Optional[str] = None
    subscription_key: Optional[str] = None
    
    # API Keys for different providers
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    
    # Additional settings
    max_tokens: Optional[int] = None
    streaming: bool = False
    
    @classmethod
    def from_env(cls, provider: Optional[str] = None) -> 'LLMConfig':
        """Create configuration from environment variables"""
        # Determine provider from env if not specified
        if provider is None:
            provider = os.getenv('LLM_PROVIDER', 'azure_openai').lower()
        
        provider_enum = LLMProvider(provider)
        
        config = cls(
            provider=provider_enum,
            temperature=float(os.getenv('LLM_TEMPERATURE', '0.1')),
            max_retries=int(os.getenv('LLM_MAX_RETRIES', '3')),
            timeout=int(os.getenv('LLM_TIMEOUT', '300')),
            streaming=os.getenv('LLM_STREAMING', 'false').lower() == 'true'
        )
        
        # Load provider-specific settings
        if provider_enum == LLMProvider.AZURE_OPENAI:
            config.azure_endpoint = os.getenv('OPENAI_BASE_URL')
            config.azure_deployment = os.getenv('AZURE_DEPLOYMENT_NAME', 'gpt-4')
            config.api_version = os.getenv('MODEL_VERSION', '2024-02-15-preview')
            config.subscription_key = os.getenv('SUBSCRIPTION_KEY')
            config.model_name = os.getenv('AZURE_MODEL_NAME', 'gpt-4')
            
        elif provider_enum == LLMProvider.OPENAI:
            config.openai_api_key = os.getenv('OPENAI_API_KEY')
            config.model_name = os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')
            config.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '4000'))
            config.enable_caching = os.getenv('LLM_ENABLE_CACHING', 'true').lower() == 'true'
            
        elif provider_enum == LLMProvider.CLAUDE:
            config.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
            config.model_name = os.getenv('CLAUDE_MODEL', 'claude-3-opus-20240229')
            config.max_tokens = int(os.getenv('CLAUDE_MAX_TOKENS', '4000'))
            config.enable_caching = os.getenv('LLM_ENABLE_CACHING', 'true').lower() == 'true'
            
        elif provider_enum == LLMProvider.GEMINI:
            config.google_api_key = os.getenv('GOOGLE_API_KEY')
            config.model_name = os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')
            config.max_tokens = int(os.getenv('GEMINI_MAX_TOKENS', '4000'))
        
        return config

class BaseLLMWrapper(ABC):
    """Abstract base class for LLM wrappers"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = None
        
    @abstractmethod
    def initialize(self):
        """Initialize the LLM connection"""
        pass
    
    @abstractmethod
    def invoke(self, messages: List[BaseMessage]) -> Any:
        """Invoke the LLM with messages"""
        pass
    
    def refresh_connection(self):
        """Refresh the LLM connection if needed"""
        self.initialize()

class AzureOpenAIWrapper(BaseLLMWrapper):
    """Wrapper for Azure OpenAI with token management"""
    
    def __init__(self, config: LLMConfig, token_manager=None, http_client=None):
        super().__init__(config)
        self.token_manager = token_manager
        self.http_client = http_client
        self.initialize()
    
    def initialize(self):
        """Initialize Azure OpenAI with existing token management"""
        try:
            # Get token if token manager is available
            initial_token = None
            if self.token_manager:
                initial_token = self.token_manager.get_valid_token()
            
            # Build kwargs for Azure OpenAI
            kwargs = {
                "temperature": self.config.temperature,
                "max_retries": self.config.max_retries,
                "request_timeout": self.config.timeout,
                "streaming": self.config.streaming,
            }
            
            # Add Azure-specific parameters
            if self.config.azure_endpoint:
                kwargs["azure_endpoint"] = self.config.azure_endpoint
            if self.config.azure_deployment:
                kwargs["azure_deployment"] = self.config.azure_deployment
            if self.config.api_version:
                kwargs["api_version"] = self.config.api_version
            if initial_token:
                kwargs["azure_ad_token"] = initial_token
            
            # Add headers if subscription key is available
            if self.config.subscription_key:
                kwargs["default_headers"] = {
                    'Ocp-Apim-Subscription-Key': self.config.subscription_key,
                    'Content-Type': 'application/json',
                }
                if initial_token:
                    kwargs["default_headers"]['Authorization'] = f"Bearer {initial_token}"
            
            # Add HTTP client if available
            if self.http_client:
                kwargs["http_client"] = self.http_client.get_client()
            
            self.llm = AzureChatOpenAI(**kwargs)
            logger.info("Azure OpenAI LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI: {e}")
            raise
    
    def invoke(self, messages: List[BaseMessage]) -> Any:
        """Invoke Azure OpenAI with automatic token refresh"""
        return self.llm.invoke(messages)
    
    def refresh_connection(self):
        """Refresh token and reinitialize if needed"""
        if self.token_manager:
            self.token_manager.force_refresh()
        self.initialize()

class OpenAIWrapper(BaseLLMWrapper):
    """Wrapper for standard OpenAI API"""
    
    def initialize(self):
        """Initialize OpenAI"""
        try:
            kwargs = {
                "model": self.config.model_name or "gpt-4-turbo-preview",
                "temperature": self.config.temperature,
                "max_retries": self.config.max_retries,
                "request_timeout": self.config.timeout,
                "streaming": self.config.streaming,
                "openai_api_key": self.config.openai_api_key,
            }
            
            if self.config.max_tokens:
                kwargs["max_tokens"] = self.config.max_tokens
            
            self.llm = ChatOpenAI(**kwargs)
            logger.info(f"OpenAI LLM initialized with model {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            raise
    
    def invoke(self, messages: List[BaseMessage]) -> Any:
        """Invoke OpenAI"""
        return self.llm.invoke(messages)


class ClaudeWrapper(BaseLLMWrapper):
    """Wrapper for Claude (Anthropic) with comprehensive error handling and attribute delegation"""

    def __init__(self, config: LLMConfig):
        """Initialize the Claude wrapper with proper error handling"""
        self.config = config
        self.llm = None

        # IMPORTANT: Call initialize() explicitly and handle any errors
        try:
            logger.info("ClaudeWrapper.__init__: Starting initialization...")
            self.initialize()
            logger.info(f"ClaudeWrapper.__init__: Completed. LLM: {self.llm is not None}")
            logger.info(f"ClaudeWrapper.__init__: Caching Status: {self.config.enable_caching}")
        except Exception as init_error:
            logger.error(f"ClaudeWrapper.__init__: Initialization failed: {init_error}")
            # Don't suppress the error - let it bubble up
            raise RuntimeError(f"ClaudeWrapper initialization failed: {init_error}") from init_error

    def initialize(self):
        """Initialize Claude with detailed error reporting"""
        try:
            # Step 1: Import check
            try:
                from langchain_anthropic import ChatAnthropic
                logger.info("langchain-anthropic package confirmed")
            except ImportError as e:
                error_msg = f"langchain-anthropic import failed: {e}"
                logger.error(error_msg)
                raise ImportError(error_msg) from e

            # Step 2: API key validation
            if not self.config.anthropic_api_key:
                error_msg = "ANTHROPIC_API_KEY is not set in environment variables"
                logger.error(error_msg)
                raise ValueError(error_msg)

            api_key = self.config.anthropic_api_key
            logger.info(f"API key loaded (length: {len(api_key)}, ends with: ...{api_key[-10:]})")

            # Step 3: Model name validation
            model_name = self.config.model_name or 'claude-3-5-sonnet-20241022'
            logger.info(f"Using model: {model_name}")

            # Step 4: Build parameters
            init_params = {
                "model": model_name,
                "anthropic_api_key": api_key,
                "temperature": self.config.temperature,
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout,
                "streaming": self.config.streaming,
            }

            # Add extended caching for claude
            if self.config.enable_caching:
                init_params["betas"] = ["extended-cache-ttl-2025-04-11"]

            # Only add max_tokens if it's set
            if self.config.max_tokens:
                init_params["max_tokens"] = self.config.max_tokens

            # Step 5: Log all parameters for debugging
            logger.info("Initializing ChatAnthropic with parameters:")
            for key, value in init_params.items():
                if key == "anthropic_api_key":
                    logger.info(f"  {key}: ...{value[-10:]}")
                else:
                    logger.info(f"  {key}: {value}")

            # Step 6: Create ChatAnthropic instance
            logger.info("Creating ChatAnthropic instance...")
            self.llm = ChatAnthropic(**init_params)

            # Step 7: Verify creation was successful
            if self.llm is None:
                raise ValueError("ChatAnthropic returned None after initialization")

            logger.info(f" ChatAnthropic created successfully: {type(self.llm)}")

            # Step 8: Test basic functionality (optional but helpful for debugging)
            try:
                logger.info("Testing Claude wrapper with simple message...")
                from langchain.schema import HumanMessage
                test_response = self.llm.invoke([HumanMessage(content="Say 'Wrapper OK'")])
                logger.info(f" Wrapper test successful: {test_response.content}")
            except Exception as test_error:
                logger.warning(f" Wrapper test failed but LLM was created: {test_error}")
                # Don't fail initialization just because test failed

            logger.info("Claude wrapper initialization completed successfully")

        except Exception as e:
            # Log the full error details
            logger.error(f" Claude wrapper initialization failed!")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error("Full traceback:", exc_info=True)

            # Set LLM to None and re-raise the original error
            self.llm = None
            raise

    def _convert_messages_for_claude(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Convert messages to Claude-compatible format.
        Claude requires at least one user message and doesn't handle system-only messages.
        """
        if not messages:
            logger.warning("Empty message list provided to Claude")
            return [HumanMessage(content="Hello")]

        converted_messages = []
        system_content = []

        # Process each message
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Collect system messages
                system_content.append(msg.content)
            else:
                # If we have system content, add it as context to a user message
                if system_content:
                    combined_system = "\n\n".join(system_content)
                    converted_messages.append(HumanMessage(
                        content=f"System context:\n{combined_system}\n\nUser request: {msg.content}"
                    ))
                    system_content = []
                else:
                    # Regular message, add as-is
                    converted_messages.append(msg)

        # Handle case where we only had system messages
        if system_content and not converted_messages:
            combined_system = "\n\n".join(system_content)
            if self.config.enable_caching:
                converted_messages.append(HumanMessage(content=combined_system, additional_kwargs={"cache_control": {"type": "ephemeral", "ttl": "1h"}}))
            else:
                converted_messages.append(HumanMessage(content=combined_system))

        # Ensure we have at least one user message
        if not any(isinstance(msg, HumanMessage) for msg in converted_messages):
            if converted_messages:
                # Convert first message to HumanMessage
                first_msg = converted_messages[0]
                if self.config.enable_caching:
                    converted_messages[0] = HumanMessage(content=str(first_msg.content), additional_kwargs={"cache_control": {"type": "ephemeral", "ttl": "1h"}})
                else:
                    converted_messages[0] = HumanMessage(content=str(first_msg.content))
            else:
                # Fallback
                converted_messages = [HumanMessage(content="Please respond.")]

        logger.debug(f"Converted {len(messages)} messages to {len(converted_messages)} Claude-compatible messages")
        return converted_messages

    def invoke(self, messages: List[BaseMessage]) -> Any:
        """Invoke Claude with message conversion"""
        if self.llm is None:
            raise ValueError("Claude LLM is not initialized. Call initialize() first.")

        try:
            # Convert messages for Claude compatibility
            converted_messages = self._convert_messages_for_claude(messages)

            # Invoke Claude
            logger.debug(f"Invoking Claude with {len(converted_messages)} messages")
            response = self.llm.invoke(converted_messages)
            logger.debug("Claude invocation successful")
            return response

        except Exception as e:
            logger.error(f"Claude invocation failed: {type(e).__name__}: {e}")

            # Provide user-friendly error messages
            error_str = str(e).lower()
            if '401' in error_str or 'authentication' in error_str:
                raise ValueError("Authentication failed. Check your ANTHROPIC_API_KEY.") from e
            elif '429' in error_str or 'rate limit' in error_str:
                raise ValueError("Rate limit exceeded. Please try again later.") from e
            elif '400' in error_str and 'message' in error_str:
                raise ValueError("Invalid message format. This might be a system-only message issue.") from e
            else:
                # Re-raise with additional context
                raise ValueError(f"Claude API error: {e}") from e

    def refresh_connection(self):
        """Refresh the LLM connection if needed"""
        logger.info("Refreshing Claude connection...")
        self.initialize()

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying LLM.
        This allows the wrapper to act as a transparent proxy for attributes like 'temperature'.
        """
        if self.llm is None:
            raise AttributeError(f"ClaudeWrapper has no attribute '{name}' and LLM is not initialized")

        try:
            return getattr(self.llm, name)
        except AttributeError:
            raise AttributeError(f"Neither ClaudeWrapper nor the underlying LLM has attribute '{name}'")

    def __setattr__(self, name, value):
        """
        Handle attribute setting - delegate to underlying LLM if not a wrapper attribute.
        """
        # These are wrapper-specific attributes that should be set on the wrapper
        wrapper_attrs = {'config', 'llm'}

        if name in wrapper_attrs or self.llm is None:
            # Set on the wrapper itself
            super().__setattr__(name, value)
        else:
            # Try to set on the underlying LLM first, fallback to wrapper
            try:
                if hasattr(self.llm, name):
                    setattr(self.llm, name, value)
                else:
                    super().__setattr__(name, value)
            except AttributeError:
                super().__setattr__(name, value)


class GeminiWrapper(BaseLLMWrapper):
    """Wrapper for Google Gemini"""
    
    def initialize(self):
        """Initialize Gemini"""
        try:
            kwargs = {
                "model": self.config.model_name or "gemini-1.5-pro",
                "temperature": self.config.temperature,
                "google_api_key": self.config.google_api_key,
                "streaming": self.config.streaming,
            }
            
            if self.config.max_tokens:
                kwargs["max_output_tokens"] = self.config.max_tokens
            
            self.llm = ChatGoogleGenerativeAI(**kwargs)
            logger.info(f"Gemini LLM initialized with model {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise
    
    def invoke(self, messages: List[BaseMessage]) -> Any:
        """Invoke Gemini"""
        return self.llm.invoke(messages)


class UniversalLLMFactory:
    """Factory class for creating LLM instances"""

    _instance = None
    _wrapper = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def create_llm(cls, config: Optional[LLMConfig] = None,
                   token_manager=None, http_client=None) -> BaseLLMWrapper:
        """
        Create an LLM wrapper based on configuration

        Args:
            config: LLM configuration (if None, loads from environment)
            token_manager: Token manager for Azure OpenAI (optional)
            http_client: HTTP client for Azure OpenAI (optional)

        Returns:
            BaseLLMWrapper instance
        """
        if config is None:
            config = LLMConfig.from_env()

        logger.info(f"Creating LLM wrapper for provider: {config.provider.value}")

        if config.provider == LLMProvider.AZURE_OPENAI:
            wrapper = AzureOpenAIWrapper(config, token_manager, http_client)
        elif config.provider == LLMProvider.OPENAI:
            wrapper = OpenAIWrapper(config)
        elif config.provider == LLMProvider.CLAUDE:
            wrapper = ClaudeWrapper(config)
        elif config.provider == LLMProvider.GEMINI:
            wrapper = GeminiWrapper(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

        # Validate that the wrapper has a valid LLM
        if wrapper.llm is None:
            raise ValueError(f"Failed to initialize LLM for provider {config.provider.value}")

        cls._wrapper = wrapper
        return wrapper

    @classmethod
    def get_current_wrapper(cls) -> Optional[BaseLLMWrapper]:
        """Get the current LLM wrapper instance"""
        return cls._wrapper

    @classmethod
    def create_langchain_compatible_llm(cls, config: Optional[LLMConfig] = None,
                                        token_manager=None, http_client=None):
        """
        Create a LangChain-compatible LLM that can be used as drop-in replacement

        IMPORTANT: For Claude, this returns the wrapper (not wrapper.llm) to handle
        message conversion for system-only messages.
        """
        try:
            wrapper = cls.create_llm(config, token_manager, http_client)

            if wrapper.llm is None:
                raise ValueError("Wrapper created but LLM is None")

            # CRITICAL FIX: For Claude, return the wrapper itself to handle message conversion
            if config and config.provider == LLMProvider.CLAUDE:
                logger.info("Returning ClaudeWrapper (not raw LLM) for message conversion support")
                return wrapper
            else:
                # For other providers, return the raw LLM
                return wrapper.llm

        except Exception as e:
            logger.error(f"Failed to create LangChain compatible LLM: {e}")
            raise


def create_universal_llm(endpoint_path="/openai4/az_openai_gpt-4o_chat"):
    """
    Backward compatible function that supports multiple providers

    IMPORTANT: For Claude, this now returns the wrapper to handle system-only messages
    """
    try:
        config = LLMConfig.from_env()

        logger.info(f"Creating universal LLM with provider: {config.provider.value}")

        # For Azure OpenAI, maintain existing token management
        if config.provider == LLMProvider.AZURE_OPENAI:
            try:
                from agent import token_manager, http_client

                # Update endpoint if provided
                if endpoint_path and config.azure_endpoint:
                    config.azure_endpoint = f"{config.azure_endpoint}{endpoint_path}"

                wrapper = UniversalLLMFactory.create_llm(config, token_manager, http_client)

                # For Azure, return the raw LLM (existing behavior)
                return wrapper.llm

            except ImportError:
                logger.warning("Could not import token_manager/http_client, creating without them")
                wrapper = UniversalLLMFactory.create_llm(config)
                return wrapper.llm

        # For Claude, return the wrapper to handle message conversion
        elif config.provider == LLMProvider.CLAUDE:
            wrapper = UniversalLLMFactory.create_llm(config)

            # Return the wrapper itself (not wrapper.llm) for Claude
            logger.info("Returning ClaudeWrapper for system message conversion support")
            return wrapper

        # For other providers, return the raw LLM
        else:
            wrapper = UniversalLLMFactory.create_llm(config)
            return wrapper.llm

    except Exception as e:
        logger.error(f"Failed to create universal LLM: {e}")
        raise