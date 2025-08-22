"""
Enhanced Claude LangChain debugging script
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_imports():
    """Test all required imports"""
    print("=== Testing Imports ===\n")
    
    try:
        import anthropic
        print(f"✅ anthropic library: {anthropic.__version__}")
    except ImportError as e:
        print(f"❌ anthropic library not found: {e}")
        return False
    
    try:
        import langchain_anthropic
        print(f"✅ langchain-anthropic library: {langchain_anthropic.__version__}")
    except ImportError as e:
        print(f"❌ langchain-anthropic library not found: {e}")
        print("Install with: pip install langchain-anthropic")
        return False
    except AttributeError:
        print("✅ langchain-anthropic library (version not accessible)")
    
    try:
        from langchain_anthropic import ChatAnthropic
        print("✅ ChatAnthropic import successful")
    except ImportError as e:
        print(f"❌ ChatAnthropic import failed: {e}")
        return False
    
    try:
        from langchain.schema import SystemMessage, HumanMessage
        print("✅ LangChain schema imports successful")
    except ImportError as e:
        print(f"❌ LangChain schema imports failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading from environment"""
    print("\n=== Testing Configuration ===\n")
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    model = os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')
    temperature = float(os.getenv('LLM_TEMPERATURE', '0.1'))
    max_tokens = int(os.getenv('CLAUDE_MAX_TOKENS', '4000'))
    
    print(f"API Key: {'✅ Found' if api_key else '❌ Missing'} ({'...'+api_key[-10:] if api_key else 'None'})")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Max Tokens: {max_tokens}")
    
    if not api_key:
        print("❌ ANTHROPIC_API_KEY is required")
        return False, None
    
    config = {
        'api_key': api_key,
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens
    }
    
    return True, config

def test_langchain_anthropic_raw(config):
    """Test ChatAnthropic initialization and basic functionality"""
    print("\n=== Testing ChatAnthropic Raw ===\n")
    
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain.schema import HumanMessage, SystemMessage
        
        print("Creating ChatAnthropic instance...")
        print(f"  Model: {config['model']}")
        print(f"  Temperature: {config['temperature']}")
        print(f"  Max Tokens: {config['max_tokens']}")
        
        llm = ChatAnthropic(
            model=config['model'],
            anthropic_api_key=config['api_key'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens'],
            max_retries=2,
            timeout=30
        )
        
        print("✅ ChatAnthropic instance created successfully")
        
        # Test 1: Simple HumanMessage
        print("\nTest 1: Simple HumanMessage...")
        try:
            response = llm.invoke([HumanMessage(content="Say 'Test 1 passed' exactly.")])
            print(f"✅ Test 1 response: {response.content}")
        except Exception as e:
            print(f"❌ Test 1 failed: {type(e).__name__}: {str(e)}")
            return False
        
        # Test 2: SystemMessage + HumanMessage
        print("\nTest 2: SystemMessage + HumanMessage...")
        try:
            response = llm.invoke([
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Say 'Test 2 passed' exactly.")
            ])
            print(f"✅ Test 2 response: {response.content}")
        except Exception as e:
            print(f"❌ Test 2 failed: {type(e).__name__}: {str(e)}")
            return False
        
        # Test 3: SystemMessage only (this might fail)
        print("\nTest 3: SystemMessage only (expected to potentially fail)...")
        try:
            response = llm.invoke([SystemMessage(content="Say 'Test 3 passed' exactly.")])
            print(f"✅ Test 3 response: {response.content}")
        except Exception as e:
            print(f"⚠️ Test 3 failed (expected): {type(e).__name__}: {str(e)}")
            print("This is the issue we need to fix in our wrapper!")
            # Don't return False here as this is expected
        
        return True
        
    except Exception as e:
        print(f"❌ ChatAnthropic test failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_config():
    """Test LLMConfig from llm_factory"""
    print("\n=== Testing LLMConfig ===\n")
    
    try:
        # Add the current directory to Python path
        sys.path.insert(0, '.')
        
        from llm_factory import LLMConfig, LLMProvider
        
        config = LLMConfig.from_env()
        
        print(f"Provider: {config.provider}")
        print(f"Model: {config.model_name}")
        print(f"Temperature: {config.temperature}")
        print(f"Max Tokens: {config.max_tokens}")
        print(f"API Key Present: {'Yes' if config.anthropic_api_key else 'No'}")
        
        if config.provider != LLMProvider.CLAUDE:
            print(f"⚠️ Warning: LLM_PROVIDER is set to {config.provider.value}, not 'claude'")
            print("Set LLM_PROVIDER=claude in your .env file")
            return False
        
        return True, config
        
    except ImportError as e:
        print(f"❌ Cannot import llm_factory: {e}")
        print("Make sure you're running this from the correct directory")
        return False, None
    except Exception as e:
        print(f"❌ LLMConfig test failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def test_claude_wrapper():
    """Test ClaudeWrapper from llm_factory"""
    print("\n=== Testing ClaudeWrapper ===\n")
    
    try:
        from llm_factory import ClaudeWrapper, LLMConfig
        from langchain.schema import SystemMessage, HumanMessage
        
        config = LLMConfig.from_env()
        
        print("Creating ClaudeWrapper...")
        wrapper = ClaudeWrapper(config)
        
        print("✅ ClaudeWrapper created successfully")
        print(f"Wrapper LLM: {wrapper.llm}")
        
        if wrapper.llm is None:
            print("❌ Wrapper LLM is None - initialization failed")
            return False
        
        # Test the wrapper's invoke method
        print("\nTesting wrapper invoke with SystemMessage...")
        try:
            response = wrapper.invoke([SystemMessage(content="Say 'Wrapper test passed' exactly.")])
            print(f"✅ Wrapper response: {response.content}")
            return True
        except Exception as e:
            print(f"❌ Wrapper invoke failed: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ ClaudeWrapper test failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_universal_factory():
    """Test the universal LLM factory"""
    print("\n=== Testing Universal LLM Factory ===\n")
    
    try:
        from llm_factory import create_universal_llm
        from langchain.schema import SystemMessage
        
        print("Creating LLM via universal factory...")
        llm = create_universal_llm()
        
        print(f"✅ Universal LLM created: {type(llm)}")
        
        # Test invocation
        print("Testing LLM invocation...")
        response = llm.invoke([SystemMessage(content="Say 'Universal factory test passed' exactly.")])
        print(f"✅ Universal LLM response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Universal factory test failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("COMPREHENSIVE CLAUDE LANGCHAIN DEBUG")
    print("=" * 60)
    
    # Test 1: Imports
    if not test_imports():
        print("\n❌ Import test failed. Fix imports before proceeding.")
        sys.exit(1)
    
    # Test 2: Configuration
    config_ok, config = test_config_loading()
    if not config_ok:
        print("\n❌ Configuration test failed. Fix environment variables.")
        sys.exit(1)
    
    # Test 3: Raw LangChain-Anthropic
    langchain_ok = test_langchain_anthropic_raw(config)
    
    # Test 4: LLMConfig
    llm_config_ok, llm_config_obj = test_llm_config()
    
    # Test 5: ClaudeWrapper (only if LLMConfig works)
    wrapper_ok = False
    if llm_config_ok:
        wrapper_ok = test_claude_wrapper()
    
    # Test 6: Universal Factory (only if wrapper works)
    factory_ok = False
    if wrapper_ok:
        factory_ok = test_universal_factory()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Imports: {'✅' if True else '❌'}")
    print(f"Configuration: {'✅' if config_ok else '❌'}")
    print(f"Raw LangChain-Anthropic: {'✅' if langchain_ok else '❌'}")
    print(f"LLMConfig: {'✅' if llm_config_ok else '❌'}")
    print(f"ClaudeWrapper: {'✅' if wrapper_ok else '❌'}")
    print(f"Universal Factory: {'✅' if factory_ok else '❌'}")
    
    if factory_ok:
        print("\n🎉 ALL TESTS PASSED! Claude integration should work.")
    elif wrapper_ok:
        print("\n⚠️ Wrapper works but factory fails. Check factory implementation.")
    elif langchain_ok:
        print("\n⚠️ Raw LangChain works but wrapper fails. Check wrapper implementation.")
    else:
        print("\n❌ Basic LangChain-Anthropic integration fails. Check installation and config.")
        
    print("\nNext steps based on results:")
    if not langchain_ok:
        print("1. Fix langchain-anthropic basic integration first")
        print("2. Check anthropic and langchain-anthropic versions:")
        print("   pip list | grep -E '(anthropic|langchain)'")
        print("3. Try upgrading: pip install --upgrade langchain-anthropic anthropic")
    elif not wrapper_ok:
        print("1. Fix ClaudeWrapper implementation")
        print("2. Check message conversion logic")
        print("3. Check initialization parameters")
    elif not factory_ok:
        print("1. Fix universal factory implementation")
        print("2. Check factory initialization logic")