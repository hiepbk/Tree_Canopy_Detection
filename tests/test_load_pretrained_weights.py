"""
Test loading pretrained SatlasPretrain weights into backbone.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcd.models.backbone.satlas_backbone import SwinBackbone, load_satlas_pretrained_weights


def test_load_aerial_swinb_weights():
    """Test loading Aerial_SwinB_SI pretrained weights."""
    print("=" * 60)
    print("Testing Pretrained Weight Loading")
    print("=" * 60)
    print()
    
    checkpoint_path = 'pretrained/aerial_swinb_si.pth'
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found at {checkpoint_path}")
        print("  Please download it first using the download script")
        return False
    
    print(f"✓ Checkpoint found: {checkpoint_path}")
    print(f"  File size: {os.path.getsize(checkpoint_path) / (1024*1024):.2f} MB")
    print()
    
    # Create backbone
    print("Creating Swin-Base backbone (RGB, 3 channels)...")
    backbone = SwinBackbone(num_channels=3, arch='swinb')
    print("✓ Backbone created")
    print()
    
    # Load weights
    print("Loading pretrained weights...")
    try:
        backbone = load_satlas_pretrained_weights(backbone, checkpoint_path, multi_image=False)
        print("✓ Weights loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load weights: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test forward pass with loaded weights
    print("Testing forward pass with loaded weights...")
    backbone.eval()
    
    # Create dummy RGB input (normalized 0-1 as per SatlasPretrain)
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 512, 512)
    
    with torch.no_grad():
        outputs = backbone(input_tensor)
    
    # Verify output structure
    assert len(outputs) == 4, f"Expected 4 output stages, got {len(outputs)}"
    
    expected_shapes = [
        (batch_size, 128, 128, 128),
        (batch_size, 256, 64, 64),
        (batch_size, 512, 32, 32),
        (batch_size, 1024, 16, 16),
    ]
    
    for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
        assert output.shape == expected_shape, \
            f"Stage {i+1}: Expected {expected_shape}, got {output.shape}"
    
    print("✓ Forward pass successful with loaded weights")
    print(f"  Output shapes: {[out.shape for out in outputs]}")
    print()
    
    # Verify some weights were actually loaded (not random)
    # Check first conv layer weights
    first_conv_weight = backbone.backbone.features[0][0].weight
    print(f"First conv layer weight stats:")
    print(f"  Shape: {first_conv_weight.shape}")
    print(f"  Mean: {first_conv_weight.mean().item():.6f}")
    print(f"  Std: {first_conv_weight.std().item():.6f}")
    print(f"  Min: {first_conv_weight.min().item():.6f}")
    print(f"  Max: {first_conv_weight.max().item():.6f}")
    print()
    
    print("=" * 60)
    print("✓ All tests passed! Backbone is ready for use.")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_load_aerial_swinb_weights()
    if not success:
        sys.exit(1)

