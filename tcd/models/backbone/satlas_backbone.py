"""
SatlasPretrain Backbone Implementation
Based on satlaspretrain_models repository
Supports loading pretrained weights for RGB aerial imagery
"""

import torch
import torch.nn as nn
import torchvision


class SwinBackbone(nn.Module):
    """
    Swin Transformer backbone for SatlasPretrain models.
    
    Supports Swin-v2-Base and Swin-v2-Tiny architectures.
    Can load pretrained weights from SatlasPretrain checkpoints.
    
    Args:
        num_channels (int): Number of input channels (3 for RGB)
        arch (str): Architecture type - 'swinb' for Swin-v2-Base or 'swint' for Swin-v2-Tiny
    """
    
    def __init__(self, num_channels=3, arch='swinb'):
        super(SwinBackbone, self).__init__()
        
        if arch == 'swinb':
            self.backbone = torchvision.models.swin_v2_b()
            self.out_channels = [
                [4, 128],   # Stage 1: stride 4, 128 channels
                [8, 256],   # Stage 2: stride 8, 256 channels
                [16, 512],  # Stage 3: stride 16, 512 channels
                [32, 1024], # Stage 4: stride 32, 1024 channels
            ]
        elif arch == 'swint':
            self.backbone = torchvision.models.swin_v2_t()
            self.out_channels = [
                [4, 96],    # Stage 1: stride 4, 96 channels
                [8, 192],   # Stage 2: stride 8, 192 channels
                [16, 384],  # Stage 3: stride 16, 384 channels
                [32, 768],  # Stage 4: stride 32, 768 channels
            ]
        else:
            raise ValueError(f"Unsupported architecture: {arch}. Use 'swinb' or 'swint'")
        
        # Replace first conv layer to accept num_channels input
        # Original Swin uses 3 channels, we need to adapt for different input channels
        original_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            num_channels,
            original_conv.out_channels,
            kernel_size=(4, 4),
            stride=(4, 4),
            bias=original_conv.bias is not None
        )
    
    def forward(self, x):
        """
        Forward pass through Swin backbone.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        
        Returns:
            list[torch.Tensor]: List of feature maps from different stages
                - Stage 1: (B, 128, H/4, W/4) for swinb or (B, 96, H/4, W/4) for swint
                - Stage 2: (B, 256, H/8, W/8) for swinb or (B, 192, H/8, W/8) for swint
                - Stage 3: (B, 512, H/16, W/16) for swinb or (B, 384, H/16, W/16) for swint
                - Stage 4: (B, 1024, H/32, W/32) for swinb or (B, 768, H/32, W/32) for swint
        """
        outputs = []
        for layer in self.backbone.features:
            x = layer(x)
            # Swin outputs (B, H, W, C), convert to (B, C, H, W)
            outputs.append(x.permute(0, 3, 1, 2))
        
        # Return features from stages: -7, -5, -3, -1 (matching SatlasPretrain output)
        return [outputs[-7], outputs[-5], outputs[-3], outputs[-1]]


class AggregationBackbone(nn.Module):
    """
    Multi-image aggregation backbone wrapper.
    
    Applies backbone to each image separately, then aggregates features
    using max pooling across temporal dimension.
    
    Args:
        num_channels (int): Number of channels per image (3 for RGB)
        backbone (nn.Module): Base backbone to apply to each image
    """
    
    def __init__(self, num_channels, backbone):
        super(AggregationBackbone, self).__init__()
        
        self.image_channels = num_channels
        self.backbone = backbone
        
        # Features from images are aggregated separately
        # Then output is concatenation across groups
        self.groups = [[0, 1, 2, 3, 4, 5, 6, 7]]
        
        ngroups = len(self.groups)
        self.out_channels = [
            (depth, ngroups * count) 
            for (depth, count) in self.backbone.out_channels
        ]
        
        self.aggregation_op = 'max'
    
    def forward(self, x):
        """
        Forward pass with multi-image aggregation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, N*C, H, W)
                where N is number of images and C is channels per image
        
        Returns:
            list[torch.Tensor]: Aggregated feature maps
        """
        # Get features of each image
        all_features = []
        for i in range(0, x.shape[1], self.image_channels):
            features = self.backbone(x[:, i:i+self.image_channels, :, :])
            all_features.append(features)
        
        # Compute aggregation over each group
        aggregated_features_list = []
        for feature_idx in range(len(all_features[0])):
            aggregated_features = []
            for group in self.groups:
                group_features = []
                for image_idx in group:
                    # Skip image indices that aren't available
                    if image_idx >= len(all_features):
                        continue
                    group_features.append(all_features[image_idx][feature_idx])
                
                # Stack group features: (N, B, C, H, W)
                group_features = torch.stack(group_features, dim=0)
                
                if self.aggregation_op == 'max':
                    group_features = torch.amax(group_features, dim=0)
                
                aggregated_features.append(group_features)
            
            # Concatenate across groups
            aggregated_features = torch.cat(aggregated_features, dim=1)
            aggregated_features_list.append(aggregated_features)
        
        return aggregated_features_list


def load_satlas_pretrained_weights(backbone, checkpoint_path, multi_image=False):
    """
    Load pretrained SatlasPretrain weights into backbone.
    
    The checkpoint has keys like 'backbone.backbone.features.0.0.weight'
    but our SwinBackbone expects 'backbone.features.0.0.weight'
    So we need to remove one 'backbone.' prefix.
    
    Args:
        backbone (nn.Module): Backbone model to load weights into
        checkpoint_path (str): Path to pretrained checkpoint file
        multi_image (bool): Whether this is a multi-image model (affects prefix handling)
    
    Returns:
        nn.Module: Backbone with loaded weights
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract backbone weights and adjust prefixes
    backbone_state_dict = {}
    for key, value in checkpoint.items():
        if 'backbone' not in key:
            continue
        
        # Handle prefix adjustment
        # Checkpoint format: 'backbone.backbone.features...' or 'backbone.features...'
        # Our model expects: 'backbone.features...'
        new_key = key
        
        # Remove one 'backbone.' prefix if it exists twice
        if new_key.startswith('backbone.backbone.'):
            # Remove first 'backbone.' to get 'backbone.features...'
            new_key = new_key.replace('backbone.backbone.', 'backbone.', 1)
        elif new_key.startswith('backbone.') and not multi_image:
            # For single-image models, keep as is if already correct
            pass
        
        backbone_state_dict[new_key] = value
    
    # Load state dict with strict=False to handle minor mismatches
    missing_keys, unexpected_keys = backbone.load_state_dict(backbone_state_dict, strict=False)
    
    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing keys when loading weights")
        if len(missing_keys) <= 10:
            print(f"  Missing: {missing_keys}")
        else:
            print(f"  Missing (first 10): {missing_keys[:10]}")
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected keys")
        if len(unexpected_keys) <= 10:
            print(f"  Unexpected: {unexpected_keys}")
        else:
            print(f"  Unexpected (first 10): {unexpected_keys[:10]}")
    
    if not missing_keys and not unexpected_keys:
        print("✓ All weights loaded successfully!")
    elif len(missing_keys) < 5 and len(unexpected_keys) < 5:
        print("✓ Most weights loaded successfully (minor mismatches expected)")
    
    return backbone

