import json
import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from llm_clients import LLMClient, GLOBAL_TEMPERATURE, GLOBAL_TOP_P, GLOBAL_MAX_TOKENS

class OpenRouterClient(LLMClient):
    """Client for OpenRouter API with free->paid model fallback and accurate cost tracking"""
    
    def __init__(self, model_name: str, config_path: str = "config/openrouter_config.json"):
        """
        Initialize OpenRouter client with model mapping and API key
        
        Args:
            model_name: Ollama model name to map to OpenRouter
            config_path: Path to OpenRouter configuration file
        """
        super().__init__(model_name)
        self.config_path = config_path
        self.config = self._load_config()
        self.api_key = self.config.get("api_key")
        self.model_mappings = self.config.get("model_mappings", {})
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not found in config file")
        
        # Get mapped models for this Ollama model name
        self.mapped_models = self.model_mappings.get(model_name, [])
        if not self.mapped_models:
            raise ValueError(f"No OpenRouter mapping found for model: {model_name}")
        
        self.current_model_index = 0  # Start with first model (usually free)
        self.base_url = "https://openrouter.ai/api/v1"
        
        # Load token costs for fallback calculations
        self.token_costs = self.config.get("token_costs", {})
        
        # Enhanced cost tracking with accurate API-based costs
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0  # Now tracks accurate API-returned costs
        self.api_calls_made = 0
        self.api_cost_calls = 0  # Calls where API returned accurate cost
        self.fallback_cost_calls = 0  # Calls where we used fallback calculation
    
    def _load_config(self) -> Dict[str, Any]:
        """Load OpenRouter configuration from JSON file"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"OpenRouter config file not found: {self.config_path}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Error loading OpenRouter config: {e}")
    
    def _get_current_model(self) -> str:
        """Get current model to use (with fallback logic)"""
        if self.current_model_index < len(self.mapped_models):
            return self.mapped_models[self.current_model_index]
        else:
            # Fallback to last model if all failed
            return self.mapped_models[-1]
    
    def _try_next_model(self) -> bool:
        """Try next model in fallback chain. Returns True if more models available."""
        self.current_model_index += 1
        return self.current_model_index < len(self.mapped_models)
    
    def _calculate_cost_from_tokens(self, current_model: str, usage: Dict[str, Any]) -> float:
        """Fallback: Calculate cost based on token counts and config pricing"""
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        # Get cost info for current model
        cost_info = self.token_costs.get(current_model)
        
        if cost_info is None:
            # No pricing info available - check if it's a free model
            if ":free" in current_model:
                return 0.0
            else:
                # Rough estimate: $0.001 per 1K tokens for paid models
                total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
                return (total_tokens / 1000) * 0.001
        
        input_cost = (input_tokens / 1000000) * cost_info["input_per_million"]
        output_cost = (output_tokens / 1000000) * cost_info["output_per_million"]
        
        return input_cost + output_cost
    
    def _make_request(self, messages: List[Dict[str, str]], max_retries: int = 2) -> str:
        """
        Make request to OpenRouter API with automatic model fallback and accurate cost tracking
        
        Args:
            messages: List of conversation messages
            max_retries: Maximum retry attempts per model
            
        Returns:
            Generated response text
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:8000",
            "X-Title": "NER Experiment System"
        }
        
        last_error = None
        models_tried = []
        
        while True:
            current_model = self._get_current_model()
            models_tried.append(current_model)
            
            # Add usage.include=true to get accurate cost information
            payload = {
                    "model": current_model,
                    "messages": messages,
                    "temperature": GLOBAL_TEMPERATURE,
                    "top_p": GLOBAL_TOP_P,
                    "max_tokens": GLOBAL_MAX_TOKENS,
                    "stream": False,
                    "usage": {"include": True}
                }

            is_grok_model = "grok" in current_model.lower() or "x-ai" in current_model.lower() \
                            or "sonoma" in current_model.lower()
            if not is_grok_model:
                payload["stop"] = ["<END_JSON>"]

            for retry in range(max_retries + 1):
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=15)
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result["choices"][0]["message"]["content"]
                        
                        # Extract usage information
                        usage = result.get("usage", {})
                        input_tokens = usage.get("prompt_tokens", 0)
                        output_tokens = usage.get("completion_tokens", 0)
                        
                        # Try to get accurate cost from API first
                        api_cost = usage.get("cost")
                        if api_cost is not None:
                            try:
                                cost = float(api_cost)
                                self.api_cost_calls += 1
                            except (ValueError, TypeError):
                                # API returned invalid cost, use fallback
                                cost = self._calculate_cost_from_tokens(current_model, usage)
                                self.fallback_cost_calls += 1
                        else:
                            # API didn't return cost, use fallback calculation
                            cost = self._calculate_cost_from_tokens(current_model, usage)
                            self.fallback_cost_calls += 1
                        
                        # Update cost tracking
                        self.total_input_tokens += input_tokens
                        self.total_output_tokens += output_tokens
                        self.total_cost += cost
                        self.api_calls_made += 1
                        
                        return content
                    
                    elif response.status_code == 429:
                        # Rate limited - try next model
                        print(f"Rate limited on {current_model}, trying next model...")
                        last_error = f"Rate limited: {response.text}"
                        break
                    
                    elif response.status_code in [400, 401, 403]:
                        # Client error - likely permanent, try next model
                        print(f"Client error on {current_model}: {response.status_code}")
                        # # >>>> FOR DEBUGGING PURPOSES <<<<
                        error_details = response.text  # 추가
                        print(f"DEBUG 400 ERROR for {current_model}: {error_details}")  # 추가
                        last_error = f"Client error {response.status_code}: {response.text}"
                        print(f"DEBUG LAST ERROR for {current_model}: {last_error}")  # 추가
                        break
                    
                    else:
                        # Server error - retry same model
                        if retry < max_retries:
                            wait_time = 2 ** retry
                            print(f"Server error on {current_model}, retrying in {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                            last_error = f"Server error {response.status_code}: {response.text}"
                            break
                
                except requests.exceptions.Timeout:
                    if retry < max_retries:
                        print(f"Timeout on {current_model}, retrying...")
                        time.sleep(2)
                        continue
                    else:
                        last_error = "Request timeout"
                        break
                
                except Exception as e:
                    if retry < max_retries:
                        print(f"Request failed on {current_model}, retrying...")
                        time.sleep(2)
                        continue
                    else:
                        last_error = str(e)
                        break
            
            # Try next model if available
            if self._try_next_model():
                print(f"Falling back to next model...")
                continue
            else:
                # No more models to try
                break
        
        # All models failed
        raise Exception(f"All OpenRouter models failed. Models tried: {models_tried}. Last error: {last_error}")
    
    def generate(self, prompt: str) -> str:
        """Generate response from a single prompt"""
        messages = [{"role": "user", "content": prompt}]
        return self._make_request(messages)
    
    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using conversation messages"""
        # Convert messages to OpenRouter format
        openrouter_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role in ["user", "assistant", "system"]:
                openrouter_messages.append({
                    "role": role,
                    "content": content
                })
        
        return self._make_request(openrouter_messages)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get enhanced usage statistics with accurate cost tracking information"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens_used": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 10),  # Accurate cost from API
            "api_calls_made": self.api_calls_made,
            "api_cost_calls": self.api_cost_calls,  # Calls with accurate API cost
            "fallback_cost_calls": self.fallback_cost_calls,  # Calls with estimated cost
            "cost_accuracy_rate": round(self.api_cost_calls / max(self.api_calls_made, 1), 10),
            "models_available": self.mapped_models,
            "current_model_index": self.current_model_index,
            "current_model": self._get_current_model(),
            "average_cost_per_call": round(self.total_cost / max(self.api_calls_made, 1), 10)
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.api_calls_made = 0
        self.api_cost_calls = 0
        self.fallback_cost_calls = 0
        self.current_model_index = 0
    
    def force_paid_model(self):
        """Force use of paid models by filtering out free models"""
        paid_models = [model for model in self.mapped_models if ":free" not in model]
        if paid_models:
            self.mapped_models = paid_models
            self.current_model_index = 0
            print(f"Forced paid models: {self.mapped_models}")
        else:
            print(f"Warning: No paid models available for {self.model_name}")