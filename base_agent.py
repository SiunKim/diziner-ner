"""
Base Agent class - MODIFIED for HF support
"""
from typing import Optional, List, Dict, Any
from llm_clients import LLMClient, OllamaClient, OpenAIClient, AnthropicClient
from hf_client import HuggingFaceClient

class BaseAgent:
    """Base class for LLM-based agents with HF and Ollama support"""
    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 ollama_base_url: str = "http://localhost:11434", 
                 hf_model_path: Optional[str] = None,
                 use_openrouter: bool = False,
                 verbose: int = 0):
        """
        Initialize Base Agent with appropriate client
        
        Args:
            model_name: Name of the model (normalized)
            api_key: API key for commercial models  
            ollama_base_url: Base URL for Ollama server
            hf_model_path: Full HF model path (if HF model)
            verbose: Verbosity level
        """
        self.model_name = model_name
        self.hf_model_path = hf_model_path
        self.verbose = verbose
        self.is_hf_model = hf_model_path is not None
        self.use_openrouter = use_openrouter
        
        if self.is_hf_model:
            self.client = HuggingFaceClient(hf_model_path)
        elif use_openrouter:
            self.client = self._initialize_openrouter_client(model_name)
        else:
            self.client = self._initialize_ollama_client(model_name, api_key, ollama_base_url)
        
        self.conversation_history = []
    
    def _initialize_openrouter_client(self, model_name: str):
        """Initialize OpenRouter client - 새 메서드"""
        from openrouter_client import OpenRouterClient
        return OpenRouterClient(model_name)

    def _initialize_ollama_client(self, model_name: str, api_key: Optional[str],
                                ollama_base_url: str) -> LLMClient:
        """Initialize appropriate LLM client for non-HF models"""
        model_lower = model_name.lower()

        if 'gpt' in model_lower and 'gpt-oss' not in model_lower:
            return OpenAIClient(model_name, api_key)
        elif 'openai' in model_lower:
            return OpenAIClient(model_name, api_key)
        elif 'claude' in model_lower or 'sonnet' in model_lower or 'opus' in model_lower:
            return AnthropicClient(model_name, api_key)
        else:
            return OllamaClient(model_name, ollama_base_url)

    def _add_to_conversation_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def _clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def _generate_with_context(self, prompt: str) -> str:
        """Generate response using conversation context"""
        self._add_to_conversation_history("user", prompt)
        
        if hasattr(self.client, 'generate_with_messages'):
            response = self.client.generate_with_messages(self.conversation_history)
        else:
            context_prompt = self._build_context_prompt()
            response = self.client.generate(context_prompt)
        
        self._add_to_conversation_history("assistant", response)
        return response

    def _build_context_prompt(self) -> str:
        """Build prompt with conversation context"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for message in self.conversation_history[:-1]:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                context_parts.append(f"Human: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")
        
        current_prompt = self.conversation_history[-1]["content"]
        context_parts.append(f"Human: {current_prompt}")
        
        return "\n\n".join(context_parts)

    def reset_conversation(self):
        """Reset conversation history"""
        self._clear_conversation_history()
        if self.verbose >= 2:
            print("Conversation history reset")