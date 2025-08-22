"""
Direct test of Claude API to bypass any wrapper issues
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_anthropic_directly():
    """Test Anthropic API directly"""
    print("=== Direct Anthropic API Test ===\n")
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    model = os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514')
    
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return False
    
    print(f"API Key: {api_key[:10]}...")
    print(f"Model: {model}")
    
    try:
        # Test with raw anthropic library
        import anthropic
        
        print("\nTesting with anthropic library directly...")
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Say 'Hello, I am working!' in exactly 5 words."}
            ]
        )
        
        print(f"✅ Direct API call successful!")
        print(f"Response: {message.content[0].text}")
        return True
        
    except Exception as e:
        print(f"❌ Direct API call failed: {type(e).__name__}: {str(e)}")
        return False

def test_langchain_anthropic_with_fix():
    """Test via langchain-anthropic with message conversion"""
    print("\n=== LangChain Anthropic Test (with fix) ===\n")
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    model = os.getenv('CLAUDE_MODEL', 'claude-sonnet-4-20250514')
    
    try:
        from langchain_anthropic import ChatAnthropic
        from langchain.schema import SystemMessage, HumanMessage
        
        print("Creating ChatAnthropic...")
        llm = ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            temperature=0.1,
            max_tokens=100
        )
        
        # Test 1: User message (should work)
        print("Test 1: User message...")
        response = llm.invoke([HumanMessage(content="Say 'Hello, I am working!' in exactly 5 words.")])
        print(f"✅ User message successful: {response.content}")
        
        # Test 2: System + User message (should work)
        print("\nTest 2: System + User message...")
        response = llm.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Say 'Hello, I am working!' in exactly 5 words.")
        ])
        print(f"✅ System + User message successful: {response.content}")
        
        # Test 3: System message only (will fail without our fix)
        print("\nTest 3: System message only (without fix - expected to fail)...")
        try:
            response = llm.invoke([SystemMessage(content="Say 'Hello, I am working!' in exactly 5 words.")])
            print(f"Response: {response.content}")
        except Exception as e:
            print(f"❌ Expected failure: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_wrapper_with_fix():
    """Test our wrapper with message conversion"""
    print("\n=== Testing Our Wrapper with Fix ===\n")
    
    try:
        from llm_factory import LLMConfig, ClaudeWrapper
        from langchain.schema import SystemMessage, HumanMessage
        
        config = LLMConfig.from_env()
        wrapper = ClaudeWrapper(config)
        
        # Test with system message only (should work with our fix)
        print("Testing with system message only...")
        messages = [SystemMessage(content="Say 'Hello, I am working!' in exactly 5 words.")]
        response = wrapper.invoke(messages)
        print(f"✅ Wrapper handled system message: {response.content}")
        
        return True
    except Exception as e:
        print(f"❌ Wrapper test failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        

if __name__ == "__main__":
    print("=" * 50)
    print("Claude Direct API Test")
    print("=" * 50)
    print()
    
    # Test direct API first
    direct_ok = test_anthropic_directly()
    
    # Then test LangChain
    langchain_ok = test_langchain_anthropic_with_fix()
    
    print("\n" + "=" * 50)
    if direct_ok and langchain_ok:
        print("✅ All tests PASSED!")
    elif direct_ok and not langchain_ok:
        print("⚠️ Direct API works but LangChain integration failed")
        print("This might be a langchain-anthropic version issue")
        print("Try: pip install --upgrade langchain-anthropic")
    else:
        print("❌ Tests FAILED - check your API key and model name")