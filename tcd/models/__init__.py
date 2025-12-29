"""
Model modules for Tree Canopy Detection.
"""

from tcd.utils import Registry

# Create registries
BACKBONE = Registry('backbone')
NECK = Registry('neck')
HEAD = Registry('head')
MODEL = Registry('model')

# Import modules (they will auto-register via decorators)
from .backbone import SwinBackbone, AggregationBackbone, load_satlas_pretrained_weights
from .neck import FPN
from .head import Mask2FormerHead
from .mask2former import Mask2Former

__all__ = [
    'BACKBONE',
    'NECK',
    'HEAD',
    'MODEL',
    'SwinBackbone',
    'AggregationBackbone',
    'load_satlas_pretrained_weights',
    'FPN',
    'Mask2FormerHead',
    'Mask2Former',
]
