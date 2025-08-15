"""
Helper functions for provider-specific adjustments
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def adjust_prompt_for_provider(prompt: str, provider: str) -> str:
    """
    Adjust prompts for specific providers if needed
    
    Some providers have different strengths or limitations
    """
    if provider == "claude":
        # Claude prefers more structured prompts
        if "```python" in prompt:
            prompt = prompt.replace("```python", "```python\n# Claude: Please ensure clean, well-commented code\n")
    
    elif provider == "gemini":
        # Gemini sometimes needs clearer instruction boundaries
        if "TASK:" not in prompt:
            prompt = f"TASK:\n{prompt}"
    
    return prompt

def adjust_response_for_provider(response: str, provider: str) -> str:
    """
    Adjust responses from specific providers if needed
    """
    if provider == "gemini":
        # Gemini sometimes adds extra markdown formatting
        if response.startswith("```") and response.count("```") % 2 != 0:
            response += "\n```"
    
    return response

def get_provider_specific_config(provider: str) -> Dict[str, Any]:
    """
    Get provider-specific configuration adjustments
    """
    configs = {
        "claude": {
            "system_prompt_style": "detailed",
            "prefers_json": True,
            "max_context_length": 200000,
        },
        "openai": {
            "system_prompt_style": "concise",
            "prefers_json": True,
            "max_context_length": 128000,
        },
        "gemini": {
            "system_prompt_style": "structured",
            "prefers_json": False,
            "max_context_length": 1000000,
        },
        "azure_openai": {
            "system_prompt_style": "standard",
            "prefers_json": True,
            "max_context_length": 128000,
        }
    }
    
    return configs.get(provider, {})