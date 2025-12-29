"""
Configuration management system.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Configuration class for managing config files.
    
    Example:
        >>> cfg = Config.fromfile('configs/model.py')
        >>> print(cfg.model)
    """
    
    def __init__(self, cfg_dict: Dict[str, Any] = None, filename: Optional[str] = None):
        """
        Args:
            cfg_dict: Config dictionary
            filename: Config file path
        """
        if cfg_dict is None:
            cfg_dict = {}
        self._cfg_dict = cfg_dict
        self._filename = filename
    
    @staticmethod
    def fromfile(filename: str, return_dict: bool = True):
        """Load config from file.
        
        Args:
            filename: Path to config file (supports .py, .yaml, .yml)
            return_dict: If True, return plain dict. If False, return Config object.
            
        Returns:
            Dict or Config object
        """
        filename = str(filename)
        if not os.path.exists(filename):
            raise FileNotFoundError(f'Config file not found: {filename}')
        
        if filename.endswith('.py'):
            cfg = Config._from_pyfile(filename)
        elif filename.endswith(('.yaml', '.yml')):
            cfg = Config._from_yaml(filename)
        else:
            raise ValueError(f'Unsupported config file format: {filename}')
        
        if return_dict:
            return cfg.to_dict()
        return cfg
    
    @staticmethod
    def _from_pyfile(filename: str) -> 'Config':
        """Load config from Python file."""
        import importlib.util
        spec = importlib.util.spec_from_file_location('config', filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        cfg_dict = {}
        for key in dir(module):
            if not key.startswith('_'):
                cfg_dict[key] = getattr(module, key)
        
        return Config(cfg_dict, filename)
    
    @staticmethod
    def _from_yaml(filename: str) -> 'Config':
        """Load config from YAML file."""
        with open(filename, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        return Config(cfg_dict, filename)
    
    def __getattr__(self, name: str):
        """Get attribute from config."""
        if name in self._cfg_dict:
            value = self._cfg_dict[name]
            if isinstance(value, dict):
                return Config(value, self._filename)
            elif isinstance(value, list):
                # Convert list items that are dicts to Config objects
                return [Config(item, self._filename) if isinstance(item, dict) else item for item in value]
            return value
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def copy(self):
        """Return a copy of the config dict."""
        return self._cfg_dict.copy()
    
    def get(self, name: str, default: Any = None):
        """Get value from config with default."""
        return self._cfg_dict.get(name, default)
    
    def __getitem__(self, name: str):
        """Get item from config."""
        return self._cfg_dict[name]
    
    def __contains__(self, name: str):
        """Check if key exists in config."""
        return name in self._cfg_dict
    
    def keys(self):
        """Get all keys."""
        return self._cfg_dict.keys()
    
    def values(self):
        """Get all values."""
        return self._cfg_dict.values()
    
    def items(self):
        """Get all items."""
        return self._cfg_dict.items()
    
    def dump(self, filename: Optional[str] = None):
        """Dump config to file."""
        if filename is None:
            filename = self._filename
        if filename is None:
            raise ValueError('filename must be provided')
        
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(filename, 'w') as f:
                yaml.dump(self._cfg_dict, f, default_flow_style=False)
        else:
            raise ValueError(f'Unsupported file format: {filename}')
    
    def to_dict(self):
        """Convert Config object to plain dict recursively."""
        def _to_dict(obj):
            if isinstance(obj, Config):
                return {k: _to_dict(v) for k, v in obj._cfg_dict.items()}
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_to_dict(item) for item in obj]
            else:
                return obj
        return _to_dict(self)
    
    def __repr__(self):
        return f'Config(filename={self._filename}, keys={list(self._cfg_dict.keys())})'

