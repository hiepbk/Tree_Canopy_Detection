"""
Training runner with hook system.
"""

from typing import List, Dict, Optional
from .hooks import Hook


class Runner:
    """Training runner that manages hooks and training state."""
    
    def __init__(
        self,
        model,
        optimizer,
        work_dir: str,
        max_epochs: int,
        rank: int = 0,
        hooks: Optional[List[Hook]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.work_dir = work_dir
        self.max_epochs = max_epochs
        self.rank = rank
        
        # Training state
        self.epoch = 0
        self.iter = 0
        self.train_losses = []
        
        # Hooks
        self.hooks = hooks or []
        self.hooks.sort(key=lambda x: x.priority)
    
    def register_hook(self, hook: Hook):
        """Register a hook."""
        self.hooks.append(hook)
        self.hooks.sort(key=lambda x: x.priority)
    
    def call_hook(self, fn_name: str, **kwargs):
        """Call hook function by name."""
        for hook in self.hooks:
            if hasattr(hook, fn_name):
                fn = getattr(hook, fn_name)
                if fn_name in ['before_train_iter', 'after_train_iter']:
                    fn(self, **kwargs)
                elif fn_name in ['before_val_epoch', 'after_val_epoch']:
                    fn(self, **kwargs)
                else:
                    fn(self)

