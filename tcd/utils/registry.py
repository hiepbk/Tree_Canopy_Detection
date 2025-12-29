"""
Registry system for models, datasets, losses, etc.
"""

from typing import Dict, Any, Optional
import inspect


class Registry:
    """
    A registry to map strings to classes.
    
    Example:
        >>> BACKBONE = Registry('backbone')
        >>> @BACKBONE.register_module()
        >>> class ResNet:
        >>>     pass
        >>> backbone = BACKBONE.build(dict(type='ResNet'))
    """
    
    def __init__(self, name: str):
        """
        Args:
            name: Registry name
        """
        self._name = name
        self._module_dict: Dict[str, type] = {}
    
    def __len__(self):
        return len(self._module_dict)
    
    def __repr__(self):
        format_str = self.__class__.__name__ + f'(name={self._name}, items={list(self._module_dict.keys())})'
        return format_str
    
    @property
    def name(self):
        return self._name
    
    @property
    def module_dict(self):
        return self._module_dict
    
    def get(self, key: str) -> Optional[type]:
        """Get the registry record.
        
        Args:
            key: The class name in string format.
            
        Returns:
            The corresponding class.
        """
        return self._module_dict.get(key, None)
    
    def register_module(self, name: Optional[str] = None, force: bool = False):
        """Register a module.
        
        A record will be added to `self._module_dict`, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.
        
        Example:
            >>> @BACKBONE.register_module()
            >>> class ResNet:
            >>>     pass
            
            >>> @BACKBONE.register_module(name='mynet')
            >>> class MyNet:
            >>>     pass
            
            >>> BACKBONE.register_module(ResNet)
        
        Args:
            name: The module name to be registered. If not specified, the class
                name will be used.
            force: Whether to override an existing class with the same name.
                Default: False.
        """
        def _register(cls):
            if not inspect.isclass(cls):
                raise TypeError(f'must be a class, but got {type(cls)}')
            module_name = name if name is not None else cls.__name__
            if not force and module_name in self._module_dict:
                raise KeyError(f'{module_name} is already registered in {self.name}')
            self._module_dict[module_name] = cls
            return cls
        
        return _register
    
    def build(self, cfg: Dict[str, Any], default_args: Optional[Dict[str, Any]] = None):
        """Build a module from config dict.
        
        Args:
            cfg: Config dict. It should at least contain the key "type".
            default_args: Default initialization arguments.
            
        Returns:
            The constructed object.
        """
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        if 'type' not in cfg:
            raise KeyError(f'cfg must contain the key "type", but got {cfg}')
        
        args = cfg.copy()
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
            if obj_cls is None:
                raise KeyError(f'{obj_type} is not in the {self.name} registry')
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(f'type must be a str or valid type, but got {type(obj_type)}')
        
        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)
        
        return obj_cls(**args)


def build_from_cfg(cfg: Dict[str, Any], registry: Registry, default_args: Optional[Dict[str, Any]] = None):
    """Build a module from config dict.
    
    Args:
        cfg: Config dict. It should at least contain the key "type".
        registry: The registry to search the type from.
        default_args: Default initialization arguments.
        
    Returns:
        The constructed object.
    """
    return registry.build(cfg, default_args)

