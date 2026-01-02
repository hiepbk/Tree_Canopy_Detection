"""
YOSO Neck: Lightweight Deformable FPN with Coordinate Features
Based on YOSO architecture for efficient panoptic segmentation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tcd.models import NECK

# Import real deformable convolution
from tcd.layers import DeformConv, ModulatedDeformConv


def get_norm(norm_type, num_channels):
    """Get normalization layer.
    
    Args:
        norm_type: 'BN', 'SyncBN', 'GN', etc.
        num_channels: Number of channels
        
    Returns:
        Normalization layer
    """
    if norm_type == 'BN' or norm_type == 'BatchNorm2d':
        return nn.BatchNorm2d(num_channels)
    elif norm_type == 'SyncBN':
        # For single GPU, fallback to BN
        try:
            from torch.nn import SyncBatchNorm
            return SyncBatchNorm(num_channels)
        except:
            return nn.BatchNorm2d(num_channels)
    elif norm_type == 'GN':
        return nn.GroupNorm(32, num_channels)
    else:
        return nn.BatchNorm2d(num_channels)


class DeformLayer(nn.Module):
    """Deformable convolution layer with upsampling."""
    
    def __init__(
        self, 
        in_planes, 
        out_planes, 
        deconv_kernel=4, 
        deconv_stride=2, 
        deconv_pad=1, 
        deconv_out_pad=0, 
        modulate_deform=True, 
        num_groups=1, 
        deform_num_groups=1, 
        dilation=1,
        norm="BN"
    ):
        super(DeformLayer, self).__init__()
        self.deform_modulated = modulate_deform
        
        # Store kernel_size for offset calculation
        self.kernel_size = 3  # DeformLayer always uses 3x3 kernel
        self.deform_num_groups = deform_num_groups
        
        if modulate_deform:
            deform_conv_op = ModulatedDeformConv
            # offset channels: 3 (offset_x, offset_y, mask) * kernel_size * kernel_size * deform_num_groups
            offset_channels = 3 * self.kernel_size * self.kernel_size * deform_num_groups
        else:
            deform_conv_op = DeformConv
            # offset channels: 2 (offset_x, offset_y) * kernel_size * kernel_size * deform_num_groups
            offset_channels = 2 * self.kernel_size * self.kernel_size * deform_num_groups
        
        
        # Offset prediction conv
        self.dcn_offset = nn.Conv2d(
            in_planes,
            offset_channels,  # Already includes deform_num_groups
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            dilation=dilation
        )
        
        # Deformable conv
        self.dcn = deform_conv_op(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            deformable_groups=deform_num_groups
        )
        
        # Initialize weights - equivalent to fvcore's c2_msra_fill
        # Real ModulatedDeformConv/DeformConv has self.weight directly
        # Override the default initialization with MSRA/Kaiming normal
        nn.init.kaiming_normal_(self.dcn.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize offset conv to zero (standard practice)
        nn.init.constant_(self.dcn_offset.weight, 0)
        nn.init.constant_(self.dcn_offset.bias, 0)
        
        # Normalization
        self.dcn_bn = get_norm(norm, out_planes)
        
        # Upsampling
        self.up_sample = nn.ConvTranspose2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=deconv_kernel,
            stride=deconv_stride, 
            padding=deconv_pad,
            output_padding=deconv_out_pad,
            bias=False
        )
        self._deconv_init()
        
        self.up_bn = get_norm(norm, out_planes)
        self.relu = nn.ReLU()
    
    def _deconv_init(self):
        """Initialize deconvolution weights with bilinear interpolation."""
        w = self.up_sample.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]
    
    def forward(self, x):
        """Forward pass."""
        
        # Ensure input is contiguous (required by deformable conv)
        out = x.contiguous()
        
        if self.deform_modulated:
            offset_mask = self.dcn_offset(out)
            
            # Split offset_mask into offset_x, offset_y, mask
            # Each has kernel_size * kernel_size * deform_num_groups channels
            channels_per_component = self.kernel_size * self.kernel_size * self.deform_num_groups
            offset_x = offset_mask[:, :channels_per_component, :, :]
            offset_y = offset_mask[:, channels_per_component:2*channels_per_component, :, :]
            mask = offset_mask[:, 2*channels_per_component:, :, :].sigmoid()
            offset = torch.cat((offset_x, offset_y), dim=1)
            
            
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("Input to dcn contains NaN/Inf")
            
            if torch.isnan(offset).any() or torch.isinf(offset).any():
                raise ValueError("Offset contains NaN/Inf")
            
            if torch.isnan(mask).any() or torch.isinf(mask).any():
                raise ValueError("Mask contains NaN/Inf")
            
            out = self.dcn(out, offset, mask)
            
            if torch.isnan(out).any() or torch.isinf(out).any():
                raise ValueError("dcn output contains NaN/Inf")
        else:
            offset = self.dcn_offset(out)
            out = self.dcn(out, offset)
        
        x = out
        x = self.dcn_bn(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.up_bn(x)
        x = self.relu(x)
        return x


class LiteDeformConv(nn.Module):
    """Lightweight Deformable FPN."""
    
    def __init__(self, in_channels, agg_dim=128, norm="BN"):
        """
        Args:
            in_channels: List of input channel numbers for each level
            agg_dim: Aggregated feature dimension (default: 128)
            norm: Normalization type ('BN', 'SyncBN', etc.)
        """
        super(LiteDeformConv, self).__init__()
        
        assert isinstance(in_channels, list) and len(in_channels) == 4, \
            "in_channels must be a list of 4 channel numbers"
        
        # Calculate output channels
        # For SwinT: [96, 192, 384, 768] -> [128, 48, 96, 192, 384]
        # For SwinB: [128, 256, 512, 1024] -> [128, 64, 128, 256, 512]
        out_channels = [agg_dim]
        for idx in range(len(in_channels)):
            if idx == 0:
                out_channels.append(in_channels[idx] // 2)
            else:
                out_channels.append(in_channels[idx - 1])
        
        # out_channels = [agg_dim, ch0//2, ch0, ch1, ch2]
        # Example SwinT: [128, 48, 96, 192, 384]
        
        # Lateral convolutions (1x1 to reduce channels)
        self.lateral_conv0 = nn.Conv2d(
            in_channels=in_channels[-1],  # Last level (stage5/stage4)
            out_channels=out_channels[-1],  # stage4 channels
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        
        self.deform_conv1 = DeformLayer(
            in_planes=out_channels[-1], 
            out_planes=out_channels[-2], 
            norm=norm
        )
        
        self.lateral_conv1 = nn.Conv2d(
            in_channels=in_channels[-2],  # stage4
            out_channels=out_channels[-2],  # stage3 channels
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        
        self.deform_conv2 = DeformLayer(
            in_planes=out_channels[-2], 
            out_planes=out_channels[-3], 
            norm=norm
        )
        
        self.lateral_conv2 = nn.Conv2d(
            in_channels=in_channels[-3],  # stage3
            out_channels=out_channels[-3],  # stage2 channels
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        
        self.deform_conv3 = DeformLayer(
            in_planes=out_channels[-3], 
            out_planes=out_channels[-4], 
            norm=norm
        )
        
        self.lateral_conv3 = nn.Conv2d(
            in_channels=in_channels[-4],  # stage2
            out_channels=out_channels[-4],  # stage2//2 channels
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        
        # Cross-Feature Aggregation (CFA)
        self.output_conv = nn.Conv2d(
            in_channels=out_channels[0],  # agg_dim
            out_channels=out_channels[0],  # agg_dim
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        
        self.bias = nn.Parameter(torch.FloatTensor(1, out_channels[0], 1, 1), requires_grad=True)
        self.bias.data.fill_(0.0)
        
        # Projection convs for CFA
        self.conv_a5 = nn.Conv2d(
            in_channels=out_channels[-1], 
            out_channels=out_channels[0], 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        self.conv_a4 = nn.Conv2d(
            in_channels=out_channels[-2], 
            out_channels=out_channels[0], 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        self.conv_a3 = nn.Conv2d(
            in_channels=out_channels[-3], 
            out_channels=out_channels[0], 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        self.conv_a2 = nn.Conv2d(
            in_channels=out_channels[-4], 
            out_channels=out_channels[0], 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
    
    def forward(self, features_list):
        """
        Forward pass.
        
        Args:
            features_list: List of 4 feature maps from backbone
                [stage2, stage3, stage4, stage5]
        
        Returns:
            Aggregated feature map [B, agg_dim, H/4, W/4]
        """
        assert len(features_list) == 4, "Expected 4 feature levels"
        
        # Top-down path with deformable convs
        # Start from stage5 (last feature)
        p5 = self.lateral_conv0(features_list[-1])
        x5 = p5
        x = self.deform_conv1(x5)  # Upsample and process
        
        # Stage4
        p4 = self.lateral_conv1(features_list[-2])
        x4 = p4 + x  # Add upsampled features
        x = self.deform_conv2(x4)  # Upsample and process
        
        # Stage3
        p3 = self.lateral_conv2(features_list[-3])
        x3 = p3 + x  # Add upsampled features
        x = self.deform_conv3(x3)  # Upsample and process
        
        # Stage2
        p2 = self.lateral_conv3(features_list[-4])
        x2 = p2 + x  # Add upsampled features
        
        # Cross-Feature Aggregation (CFA)
        # Upsample all features to stage2 resolution and aggregate
        x5 = F.interpolate(
            self.conv_a5(x5), 
            scale_factor=8, 
            align_corners=False, 
            mode='bilinear'
        )
        x4 = F.interpolate(
            self.conv_a4(x4), 
            scale_factor=4, 
            align_corners=False, 
            mode='bilinear'
        )
        x3 = F.interpolate(
            self.conv_a3(x3), 
            scale_factor=2, 
            align_corners=False, 
            mode='bilinear'
        )
        x2 = self.conv_a2(x2)
        
        # Element-wise addition
        x = x5 + x4 + x3 + x2 + self.bias
        
        # Final output conv
        x = self.output_conv(x)
        
        return x


@NECK.register_module()
class YOSONeck(nn.Module):
    """YOSO Neck: Lightweight Deformable FPN with Coordinate Features."""
    
    def __init__(
        self,
        in_channels: list,
        agg_dim: int = 128,
        hidden_dim: int = 256,
        norm: str = "BN",
    ):
        """
        Args:
            in_channels: List of input channel numbers for each level
            agg_dim: Aggregated feature dimension (default: 128)
            hidden_dim: Hidden dimension for output (default: 256)
            norm: Normalization type ('BN', 'SyncBN', etc.)
        """
        super(YOSONeck, self).__init__()
        
        assert isinstance(in_channels, list) and len(in_channels) == 4, \
            "in_channels must be a list of 4 channel numbers"
        
        self.agg_dim = agg_dim
        self.hidden_dim = hidden_dim
        
        # LiteDeformConv FPN
        self.deconv = LiteDeformConv(
            in_channels=in_channels,
            agg_dim=agg_dim,
            norm=norm
        )
        
        # Coordinate feature + localization conv
        self.loc_conv = nn.Conv2d(
            in_channels=agg_dim + 2,  # +2 for coordinate features
            out_channels=hidden_dim,
            kernel_size=1,
            stride=1
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_coord(self, input_feat):
        """Generate normalized coordinate features.
        
        Args:
            input_feat: Input feature tensor [B, C, H, W]
        
        Returns:
            Coordinate features [B, 2, H, W] with values in [-1, 1]
        """
        x_range = torch.linspace(-1, 1, input_feat.shape[-1], device=input_feat.device)
        y_range = torch.linspace(-1, 1, input_feat.shape[-2], device=input_feat.device)
        y, x = torch.meshgrid(y_range, x_range, indexing='ij')
        y = y.expand([input_feat.shape[0], 1, -1, -1])
        x = x.expand([input_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        return coord_feat
    
    def forward(self, inputs: list) -> list:
        """
        Forward pass.
        
        Args:
            inputs: List of feature maps from backbone
                [stage2, stage3, stage4, stage5]
        
        Returns:
            List containing single feature map [B, hidden_dim, H/4, W/4]
            (Returns as list to match head interface)
        """
        assert len(inputs) == 4, "Expected 4 feature levels from backbone"
        
        # LiteDeformConv FPN
        features = self.deconv(inputs)  # [B, agg_dim, H/4, W/4]
        
        # Add coordinate features
        coord_feat = self.generate_coord(features)  # [B, 2, H/4, W/4]
        features = torch.cat([features, coord_feat], 1)  # [B, agg_dim+2, H/4, W/4]
        
        # Final localization conv
        features = self.loc_conv(features)  # [B, hidden_dim, H/4, W/4]
        
        # Return as list to match head interface
        return [features]

