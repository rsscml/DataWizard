"""
Test the fixed ClaudeWrapper
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Setup logging to see detailed output
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

# Load environment
load_dotenv()

def test_fixed_wrapper():
    """Test the fixed ClaudeWrapper implementation"""
    print("=== Testing Fixed ClaudeWrapper ===\n")
    
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        # Import the factory components
        from llm_factory import LLMConfig, ClaudeWrapper
        from langchain.schema import SystemMessage, HumanMessage
        
        print("1. Creating LLMConfig...")
        config = LLMConfig.from_env()
        print(f"   ✅ Config loaded - Provider: {config.provider}, Model: {config.model_name}")
        
        print("\n2. Creating ClaudeWrapper...")
        wrapper = ClaudeWrapper(config)
        
        print(f"\n3. Checking wrapper.llm: {wrapper.llm}")
        if wrapper.llm is None:
            print("   ❌ wrapper.llm is still None - check the error logs above")
            return False
        else:
            print("   ✅ wrapper.llm created successfully")
        
        print("\n4. Testing wrapper invoke with HumanMessage...")
        response = wrapper.invoke([HumanMessage(content="Say 'Fixed wrapper test passed'")])
        print(f"   ✅ HumanMessage test: {response.content}")
        
        print("\n5. Testing wrapper invoke with SystemMessage (should work now)...")
        response = wrapper.invoke([SystemMessage(content="Say 'System message conversion works'")])
        print(f"   ✅ SystemMessage test: {response.content}")
        
        print("\n6. Testing wrapper invoke with System + Human messages...")
        response = wrapper.invoke([
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Say 'Combined messages work'")
        ])
        print(f"   ✅ Combined messages test: {response.content}")
        
        print("\n🎉 All wrapper tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Wrapper test failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_universal_factory():
    """Test the universal factory after fixing the wrapper"""
    print("\n=== Testing Universal Factory with Fixed Wrapper ===\n")
    
    try:
        from llm_factory import create_universal_llm
        from langchain.schema import SystemMessage
        
        print("Creating LLM via universal factory...")
        llm = create_universal_llm()
        
        print(f"✅ Universal LLM created: {type(llm)}")
        
        # Test with system message (this should work now)
        print("Testing universal LLM with SystemMessage...")
        response = llm.invoke([SystemMessage(content="Say 'Universal factory works with Claude'")])
        print(f"✅ Universal factory test: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Universal factory test failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def debug_current_wrapper():
    """Debug the current wrapper to see what's failing"""
    print("=== Debugging Current Wrapper ===\n")
    
    try:
        from llm_factory import LLMConfig, ClaudeWrapper
        
        config = LLMConfig.from_env()
        print(f"Config: {config.provider}, {config.model_name}")
        
        # Create wrapper but catch initialization errors
        print("Creating wrapper...")
        wrapper = ClaudeWrapper(config)
        
        # Let's manually call initialize to see the error
        print("Calling initialize manually...")
        wrapper.initialize()
        
        print(f"After manual initialize: {wrapper.llm}")
        
    except Exception as e:
        print(f"Manual initialize failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING FIXED CLAUDE WRAPPER")
    print("=" * 60)
    
    # First, let's debug what's wrong with the current wrapper
    debug_current_wrapper()
    
    print("\n" + "=" * 60)
    print("NOW TESTING WITH FIXES")
    print("=" * 60)
    
    # Test the fixed wrapper
    wrapper_ok = test_fixed_wrapper()
    
    # Test universal factory if wrapper works
    if wrapper_ok:
        factory_ok = test_universal_factory()
        
        if factory_ok:
            print("\n🎉 ALL TESTS PASSED! Claude integration is working!")
        else:
            print("\n⚠️ Wrapper works but factory needs adjustment")
    else:
        print("\n❌ Wrapper still has issues - check the error messages above")
        print("\nCommon fixes:")
        print("1. Make sure your .env has: LLM_PROVIDER=claude")
        print("2. Check ANTHROPIC_API_KEY format")
        print("3. Verify model name: claude-sonnet-4-20250514")
        print("4. Try upgrading: pip install --upgrade langchain-anthropic")