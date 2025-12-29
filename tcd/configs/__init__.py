"""
Configuration modules.
"""

# Import config module
try:
    from . import tcd_config
    __all__ = ['tcd_config']
except ImportError:
    __all__ = []
