"""
Test script to verify LLM initialization
"""
import os
import logging
from llm_factory import LLMConfig, UniversalLLMFactory, LLMProvider
from langchain.schema import SystemMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llm_initialization():
    """Test LLM initialization for the configured provider"""
    try:
        # Load configuration
        config = LLMConfig.from_env()
        print(f"Testing LLM provider: {config.provider.value}")
        print(f"Model: {config.model_name}")
        
        # Check API key
        if config.provider == LLMProvider.CLAUDE:
            if not config.anthropic_api_key:
                print("ERROR: ANTHROPIC_API_KEY is not set!")
                return False
            print(f"API Key present: Yes (length: {len(config.anthropic_api_key)})")
        
        # Create LLM
        print("Creating LLM wrapper...")
        wrapper = UniversalLLMFactory.create_llm(config)
        
        if wrapper.llm is None:
            print("ERROR: LLM is None after initialization!")
            return False
        
        print("LLM created successfully!")
        
        # Test invocation
        print("Testing LLM invocation...")
        test_message = SystemMessage(content="Say 'Hello, I am working!' in 5 words or less.")
        response = wrapper.invoke([test_message])
        print(f"Response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if test_llm_initialization():
        print("\nLLM initialization test PASSED!")
    else:
        print("\nLLM initialization test FAILED!")