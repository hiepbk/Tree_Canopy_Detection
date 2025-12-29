"""
Feature Pyramid Network (FPN) for multi-scale feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tcd.models import NECK


@NECK.register_module()
class FPN(nn.Module):
    """
    Feature Pyramid Network.
    
    Args:
        in_channels: List of input channel numbers for each level
        out_channels: Output channel number for each level
        num_outs: Number of output levels
        start_level: Index of the start input backbone level
        add_extra_convs: Whether to add extra conv layers
    """
    
    def __init__(
        self,
        in_channels: list,
        out_channels: int = 256,
        num_outs: int = 4,
        start_level: int = 0,
        add_extra_convs: str = 'on_input',
    ):
        super(FPN, self).__init__()
        
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.add_extra_convs = add_extra_convs
        
        # Lateral convs
        self.lateral_convs = nn.ModuleList()
        # FPN convs
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.start_level, self.num_ins):
            l_conv = nn.Conv2d(
                in_channels[i],
                out_channels,
                1,
            )
            fpn_conv = nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
            )
            
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        
        # Add extra levels
        extra_levels = num_outs - self.num_ins + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[-1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                )
                self.fpn_convs.append(extra_fpn_conv)
    
    def forward(self, inputs: list) -> list:
        """Forward function.
        
        Args:
            inputs: List of feature maps from backbone
            
        Returns:
            List of FPN feature maps
        """
        assert len(inputs) == len(self.in_channels)
        
        # Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # Upsample and add
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode='nearest',
            )
        
        # Build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]
        
        # Add extra levels
        if self.num_outs > len(fpn_outs):
            # Use max pool to get more levels on top of outputs
            for i in range(self.num_outs - used_backbone_levels):
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.num_ins - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = fpn_outs[-1]
                else:
                    raise NotImplementedError
                
                fpn_outs.append(self.fpn_convs[used_backbone_levels + i](extra_source))
                if i < self.num_outs - used_backbone_levels - 1:
                    fpn_outs[-1] = F.max_pool2d(fpn_outs[-1], kernel_size=1, stride=2, padding=0)
        
        return fpn_outs

