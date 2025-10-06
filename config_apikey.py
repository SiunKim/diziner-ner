import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

class Config:
    """Centralized configuration management for the supervisor system"""
    
    def __init__(self, config_file: str = "openai_apikey.json"):
        self.config_file = config_file
        self._config_data = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        config_path = Path(self.config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config_data = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
                self._config_data = {}
        else:
            # Create default config file
            self.create_default_config()
    
    def create_default_config(self):
        """Create a default configuration file"""
        default_config = {
            "openai": {
                "api_key": "",
                "default_model": "gpt-5-2025-08-07"
            },
            "supervisor": {
                "enable_cost_estimation": True,
                "enable_human_validation": True,
                "verbose": 1,
                "default_output_dir": "supervisor_output"
            },
            "model_pricing": {
                "gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.01},
                "gpt-4o-mini-2024-07-18": {"input": 0.00015, "output": 0.0006},
                "gpt-4-turbo-2024-04-09": {"input": 0.01, "output": 0.03},
                "gpt-5": {"input": 0.00125, "output": 0.01},
                "gpt-5-2025-08-07": {"input": 0.00125, "output": 0.01}
            }
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            print(f"Created default config file: {self.config_file}")
            print("Please add your OpenAI API key to the config file.")
            self._config_data = default_config
        except Exception as e:
            print(f"Could not create config file: {e}")
            self._config_data = default_config
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from config or environment"""
        # First try config file
        api_key = self._config_data.get("openai", {}).get("api_key", "")
        
        if api_key:
            return api_key
        
        # Fall back to environment variable
        api_key = os.getenv('OPENAI_API_KEY', '')
        
        if not api_key:
            print("Warning: No OpenAI API key found in config file or environment variable")
            print("Please set OPENAI_API_KEY environment variable or add it to openai_apikey.json")
        
        return api_key if api_key else None
    
    def get_model_pricing(self, model_name: str) -> tuple[float, float]:
        """Get model pricing from config"""
        pricing = self._config_data.get("model_pricing", {}).get(model_name, {})
        
        if pricing:
            return pricing["input"], pricing["output"]
        
        # Default to gpt-4o pricing
        default_pricing = self._config_data.get("model_pricing", {}).get("gpt-4o-2024-11-20", {})
        if default_pricing:
            print(f"Warning: Pricing not found for {model_name}, using default pricing")
            return default_pricing["input"], default_pricing["output"]
        
        # Hard-coded fallback
        print(f"Warning: No pricing found for {model_name}, using fallback pricing")
        return 0.0025, 0.01
    
    def get_supervisor_config(self) -> Dict[str, Any]:
        """Get supervisor configuration"""
        return self._config_data.get("supervisor", {
            "enable_cost_estimation": True,
            "enable_human_validation": True,
            "verbose": 1,
            "default_output_dir": "supervisor_output"
        })
    
    def get_default_model(self) -> str:
        """Get default OpenAI model"""
        return self._config_data.get("openai", {}).get("default_model", "gpt-4o-2024-11-20")

# Global config instance
config = Config()
