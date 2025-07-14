"""
Configuration loading utilities.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Load and manage configuration files."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_file: str = "config.yaml") -> Dict[str, Any]:
        """Load main configuration file."""
        config_path = self.config_dir / config_file
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Replace environment variables
        config = self._replace_env_vars(config)
        
        return config
    
    def load_model_config(self, model_config_file: str = "model_config.yaml") -> Dict[str, Any]:
        """Load model-specific configuration."""
        return self.load_config(model_config_file)
    
    def _replace_env_vars(self, config: Any) -> Any:
        """Replace ${VAR} patterns with environment variables."""
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        else:
            return config


def load_config(config_file: str = "config.yaml") -> Dict[str, Any]:
    """Convenience function to load configuration."""
    loader = ConfigLoader()
    return loader.load_config(config_file)