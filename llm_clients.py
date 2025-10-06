import json
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Global inference parameters
GLOBAL_TEMPERATURE = 0.0
GLOBAL_TOP_P = 1.0
GLOBAL_REPEAT_PENALTY = 1.0
GLOBAL_FREQUENCY_PENALTY = 0.0
GLOBAL_PRESENCE_PENALTY = 0.0
GLOBAL_MAX_TOKENS = 8000 # 16000 → 4000
PREFER_PAID_MODELS = True

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from a single prompt"""
        pass
    
    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response using conversation messages
        Default implementation converts messages to single prompt
        """
        # Convert messages to single prompt for clients without native message support
        prompt_parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        full_prompt = "\n\n".join(prompt_parts)
        return self.generate(full_prompt)

class OllamaClient(LLMClient):
    """Client for Ollama local models"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = base_url.rstrip('/')
    
    def generate(self, prompt: str) -> str:
        """Generate response using Ollama API"""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": GLOBAL_TEMPERATURE,
                "top_p": GLOBAL_TOP_P,
                "repeat_penalty": GLOBAL_REPEAT_PENALTY,
                "num_predict": GLOBAL_MAX_TOKENS
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=20)
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse Ollama response: {str(e)}")
    
    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response using Ollama chat API if available, 
        otherwise fall back to prompt concatenation
        """
        # Try using Ollama's chat API first
        url = f"{self.base_url}/api/chat"
        
        # Convert our message format to Ollama's expected format
        ollama_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Ollama uses "user" and "assistant" roles
            if role in ["user", "assistant"]:
                ollama_messages.append({
                    "role": role,
                    "content": content
                })
        
        payload = {
            "model": self.model_name,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": GLOBAL_TEMPERATURE,
                "top_p": GLOBAL_TOP_P,
                "repeat_penalty": GLOBAL_REPEAT_PENALTY,
                "num_predict": GLOBAL_MAX_TOKENS
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=20)
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except requests.exceptions.RequestException:
            # If chat API fails, fall back to concatenated prompt approach
            return super().generate_with_messages(messages)
        except json.JSONDecodeError:
            # If JSON parsing fails, fall back to concatenated prompt approach
            return super().generate_with_messages(messages)

class OpenAIClient(LLMClient):
    """Client for OpenAI GPT models"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        if not api_key:
            raise ValueError("OpenAI API key is required")
    
    def generate(self, prompt: str) -> str:
        """Generate response using OpenAI API"""
        import openai
        
        openai.api_key = self.api_key
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=GLOBAL_TEMPERATURE,
                top_p=GLOBAL_TOP_P,
                frequency_penalty=GLOBAL_FREQUENCY_PENALTY,
                presence_penalty=GLOBAL_PRESENCE_PENALTY,
                max_tokens=GLOBAL_MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using OpenAI chat API with message history"""
        import openai
        
        openai.api_key = self.api_key
        
        # Convert our message format to OpenAI's expected format
        openai_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # OpenAI uses "user", "assistant", and "system" roles
            if role in ["user", "assistant", "system"]:
                openai_messages.append({
                    "role": role,
                    "content": content
                })
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=GLOBAL_TEMPERATURE,
                top_p=GLOBAL_TOP_P,
                frequency_penalty=GLOBAL_FREQUENCY_PENALTY,
                presence_penalty=GLOBAL_PRESENCE_PENALTY,
                max_tokens=GLOBAL_MAX_TOKENS
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

class AnthropicClient(LLMClient):
    """Client for Anthropic Claude models"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        if not api_key:
            raise ValueError("Anthropic API key is required")
    
    def generate(self, prompt: str) -> str:
        """Generate response using Anthropic API"""
        import anthropic
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        try:
            response = client.messages.create(
                model=self.model_name,
                max_tokens=GLOBAL_MAX_TOKENS,
                temperature=GLOBAL_TEMPERATURE,
                top_p=GLOBAL_TOP_P,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using Anthropic API with message history"""
        import anthropic
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        # Convert our message format to Anthropic's expected format
        anthropic_messages = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            # Anthropic uses "user" and "assistant" roles
            if role in ["user", "assistant"]:
                anthropic_messages.append({
                    "role": role,
                    "content": content
                })
        
        try:
            response = client.messages.create(
                model=self.model_name,
                max_tokens=GLOBAL_MAX_TOKENS,
                temperature=GLOBAL_TEMPERATURE,
                top_p=GLOBAL_TOP_P,
                messages=anthropic_messages
            )
            
            return response.content[0].text
            
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")

class OpenRouterClient(LLMClient):
    """Enhanced OpenRouter client with external fallback control and improved rate limit handling"""
    
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
        if PREFER_PAID_MODELS:
            self.current_model_index = len(self.mapped_models) - 1
        else:
            self.current_model_index = 0 
        self.base_url = "https://openrouter.ai/api/v1"

        self.error_tracking = {
                    'timeout_errors': 0,
                    'rate_limit_errors': 0,
                    'client_errors': 0,
                    'server_errors': 0,
                    'unknown_errors': 0
                }
        self.request_times = []
        
        # Keep token_costs for fallback (when API doesn't return cost)
        self.token_costs = self.config.get("token_costs", {})
        
        # Enhanced cost tracking with accurate API-based costs
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0  # Now tracks accurate API-returned costs
        self.api_calls_made = 0
        self.api_cost_calls = 0  # Calls where API returned accurate cost
        self.fallback_cost_calls = 0  # Calls where we used fallback calculation
        
        self.rate_limit_hits = {}  # {model_name: [timestamp1, timestamp2, ...]}
        self.recent_errors = {}  # {model_name: {'count': int, 'last_error': timestamp}}
        self._rate_limit_window = 300  # 5 minutes window for rate limit tracking
    
    def _load_config(self) -> Dict[str, Any]:
        """Load OpenRouter configuration from JSON file"""
        try:
            from pathlib import Path
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
            return self.mapped_models[-1]
    
    def _try_next_model(self) -> bool:
        """Try next model in fallback chain. Returns True if more models available."""
        self.current_model_index += 1
        return self.current_model_index < len(self.mapped_models)
    
    def _track_rate_limit(self, model_name: str):
        """Track rate limit hit for a model"""
        current_time = time.time()
        if model_name not in self.rate_limit_hits:
            self.rate_limit_hits[model_name] = []
        
        self.rate_limit_hits[model_name].append(current_time)
        
        # Clean old entries outside the window
        self.rate_limit_hits[model_name] = [
            t for t in self.rate_limit_hits[model_name] 
            if current_time - t < self._rate_limit_window
        ]
    
    def _is_model_rate_limited(self, model_name: str, threshold: int = 3) -> bool:
        """Check if model is likely rate limited based on recent history"""
        if model_name not in self.rate_limit_hits:
            return False
        
        current_time = time.time()
        recent_hits = [
            t for t in self.rate_limit_hits[model_name]
            if current_time - t < self._rate_limit_window
        ]
        
        return len(recent_hits) >= threshold
    
    def force_paid_model(self):
        """Force client to use paid model (skip free tier)"""
        if len(self.mapped_models) > 1:
            self.current_model_index = len(self.mapped_models) - 1
    
    def reset_to_free_model(self):
        """Reset client to try free model first"""
        self.current_model_index = 0
    
    def get_current_model_tier(self) -> str:
        """Get current model tier (free/paid)"""
        current_model = self._get_current_model()
        return "free" if ":free" in current_model or self.current_model_index == 0 else "paid"
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get rate limit status for all models"""
        status = {}
        for model in self.mapped_models:
            status[model] = {
                'is_rate_limited': self._is_model_rate_limited(model),
                'recent_hits': len(self.rate_limit_hits.get(model, [])),
                'tier': 'free' if ':free' in model else 'paid'
            }
        return status
    
    def _calculate_cost_from_tokens(self, current_model: str, usage: Dict[str, Any]) -> float:
        """Fallback: Calculate cost based on token counts and config pricing"""
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        # Get cost info for current model
        cost_info = self.token_costs.get(current_model)
        
        if cost_info is None:
            # Check if it's a free model
            if ":free" in current_model:
                return 0.0
            else:
                # Rough estimate for paid models
                total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
                return (total_tokens / 1000000) * 0.1  # $0.1 per million tokens fallback
        
        input_cost = (input_tokens / 1000000) * cost_info["input_per_million"]
        output_cost = (output_tokens / 1000000) * cost_info["output_per_million"]
        
        return input_cost + output_cost
    
    def _make_request(self, messages: List[Dict[str, str]], max_retries: int = 2) -> str:
        """Make request to OpenRouter API with enhanced fallback and rate limit management"""
        import time
        
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:8000",
            "X-Title": "NER Experiment System"
        }
        
        last_error = None
        models_tried = []
        original_model_index = self.current_model_index
        
        while True:
            current_model = self._get_current_model()
            models_tried.append(current_model)
            
            # Skip models that are recently rate limited
            if self._is_model_rate_limited(current_model):
                if self._try_next_model():
                    continue
                else:
                    # No more models, reset and try anyway
                    self.current_model_index = original_model_index
                    current_model = self._get_current_model()
            
            # Add usage.include=true to get accurate cost information
            payload = {
                "model": current_model,
                "messages": messages,
                "temperature": GLOBAL_TEMPERATURE,
                "top_p": GLOBAL_TOP_P,
                "max_tokens": GLOBAL_MAX_TOKENS,
                "stream": False,
                "usage": {"include": True}  # Request cost information
            }
            
            for retry in range(max_retries + 1):
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=20)
                    if response.status_code == 429:
                        self.error_tracking['rate_limit_errors'] += 1
                    elif response.status_code in [400, 401, 403]:
                        self.error_tracking['client_errors'] += 1
                        # # >>>> FOR DEBUGGING PURPOSES <<<<
                        # error_details = response.text
                        # print(f"DEBUG 400 ERROR for {current_model}: {error_details}")
                        # last_error = f"Client error {response.status_code}: {response.text}"
                        break
                    elif response.status_code >= 500:
                        self.error_tracking['server_errors'] += 1
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result["choices"][0]["message"]["content"]
                        
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
                        
                        # Update statistics
                        self.total_input_tokens += input_tokens
                        self.total_output_tokens += output_tokens
                        self.total_cost += cost
                        self.api_calls_made += 1
                        
                        return content
                    
                    elif response.status_code == 429:
                        # Rate limited - track and try next model
                        self._track_rate_limit(current_model)
                        last_error = f"Rate limited: {response.text}"
                        break
                    
                    elif response.status_code in [400, 401, 403]:
                        # Client error - try next model
                        last_error = f"Client error {response.status_code}: {response.text}"
                        break
                    
                    else:
                        # Server error - retry same model
                        if retry < max_retries:
                            wait_time = 2 ** retry
                            time.sleep(wait_time)
                            continue
                        else:
                            last_error = f"Server error {response.status_code}: {response.text}"
                            break
                
                except requests.exceptions.Timeout:            
                    self.error_tracking['timeout_errors'] += 1
                    if retry < max_retries:
                        time.sleep(2)
                        continue
                    else:
                        last_error = "Request timeout"
                        break
                
                except Exception as e:
                    self.error_tracking['unknown_errors'] += 1
                    if retry < max_retries:
                        time.sleep(2)
                        continue
                    else:
                        last_error = str(e)
                        break
            
            # Try next model if available
            if self._try_next_model():
                continue
            else:
                break
        
        # All models failed
        raise Exception(f"All OpenRouter models failed. Models tried: {models_tried}. Last error: {last_error}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """get_error_statistics"""
        total_errors = sum(self.error_tracking.values())
        return {
            'error_breakdown': self.error_tracking.copy(),
            'total_errors': total_errors,
            'error_rate': total_errors / max(self.api_calls_made, 1),
            'avg_request_time': sum(self.request_times) / max(len(self.request_times), 1) if self.request_times else 0
        }
    
    def generate(self, prompt: str) -> str:
        """Generate response from a single prompt"""
        messages = [{"role": "user", "content": prompt}]
        return self._make_request(messages)
    
    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """Generate response using conversation messages"""
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
        """Get enhanced usage statistics with rate limit information"""
        base_stats = {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens_used": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 6),
            "api_calls_made": self.api_calls_made,
            "api_cost_calls": self.api_cost_calls,
            "fallback_cost_calls": self.fallback_cost_calls,
            "cost_accuracy_rate": round(self.api_cost_calls / max(self.api_calls_made, 1), 3),
            "models_available": self.mapped_models,
            "current_model_index": self.current_model_index,
            "current_model": self._get_current_model(),
            "current_model_tier": self.get_current_model_tier(),
            "average_cost_per_call": round(self.total_cost / max(self.api_calls_made, 1), 6)
        }
        
        # Add rate limit information
        base_stats["rate_limit_status"] = self.get_rate_limit_status()
        base_stats["total_rate_limit_hits"] = sum(len(hits) for hits in self.rate_limit_hits.values())
        
        return base_stats

    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.api_calls_made = 0
        self.api_cost_calls = 0
        self.fallback_cost_calls = 0
        self.current_model_index = 0
        # Don't reset rate limit tracking - keep for better decisions

def create_llm_client(model_name: str, api_key: Optional[str] = None, 
                     ollama_base_url: str = "http://localhost:11434",
                     use_openrouter: bool = False) -> LLMClient:
    """
    Factory function to create appropriate LLM client based on model name
    
    Args:
        model_name: Name of the model
        api_key: API key for commercial models
        ollama_base_url: Base URL for Ollama server
        use_openrouter: Whether to use OpenRouter instead of other clients
        
    Returns:
        Appropriate LLMClient instance
    """
    if use_openrouter:
        return OpenRouterClient(model_name)
    
    model_lower = model_name.lower()
    
    if 'gpt' in model_lower or 'openai' in model_lower:
        return OpenAIClient(model_name, api_key)
    elif 'claude' in model_lower or 'sonnet' in model_lower or 'opus' in model_lower:
        return AnthropicClient(model_name, api_key)
    else:
        return OllamaClient(model_name, ollama_base_url)