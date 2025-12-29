"""
Unit tests for SatlasPretrain backbone implementation.
Tests backbone creation, forward pass, and weight loading.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcd.models.backbone.satlas_backbone import SwinBackbone, AggregationBackbone, load_satlas_pretrained_weights


class TestSwinBackbone:
    """Test SwinBackbone implementation."""
    
    def test_swinb_initialization(self):
        """Test Swin-Base backbone initialization."""
        backbone = SwinBackbone(num_channels=3, arch='swinb')
        
        assert backbone is not None
        assert len(backbone.out_channels) == 4
        assert backbone.out_channels[0] == [4, 128]
        assert backbone.out_channels[-1] == [32, 1024]
        print("✓ Swin-Base initialization successful")
    
    def test_swint_initialization(self):
        """Test Swin-Tiny backbone initialization."""
        backbone = SwinBackbone(num_channels=3, arch='swint')
        
        assert backbone is not None
        assert len(backbone.out_channels) == 4
        assert backbone.out_channels[0] == [4, 96]
        assert backbone.out_channels[-1] == [32, 768]
        print("✓ Swin-Tiny initialization successful")
    
    def test_swinb_forward_pass(self):
        """Test Swin-Base forward pass with RGB input."""
        backbone = SwinBackbone(num_channels=3, arch='swinb')
        backbone.eval()
        
        # Create dummy RGB input: (batch, channels, height, width)
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 512, 512)
        
        with torch.no_grad():
            outputs = backbone(input_tensor)
        
        # Check output structure
        assert len(outputs) == 4, f"Expected 4 output stages, got {len(outputs)}"
        
        # Check output shapes for Swin-Base
        expected_shapes = [
            (batch_size, 128, 128, 128),   # Stage 1: H/4, W/4
            (batch_size, 256, 64, 64),     # Stage 2: H/8, W/8
            (batch_size, 512, 32, 32),     # Stage 3: H/16, W/16
            (batch_size, 1024, 16, 16),    # Stage 4: H/32, W/32
        ]
        
        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            assert output.shape == expected_shape, \
                f"Stage {i+1}: Expected {expected_shape}, got {output.shape}"
        
        print("✓ Swin-Base forward pass successful")
        print(f"  Output shapes: {[out.shape for out in outputs]}")
    
    def test_swint_forward_pass(self):
        """Test Swin-Tiny forward pass with RGB input."""
        backbone = SwinBackbone(num_channels=3, arch='swint')
        backbone.eval()
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 512, 512)
        
        with torch.no_grad():
            outputs = backbone(input_tensor)
        
        assert len(outputs) == 4
        
        # Check output shapes for Swin-Tiny
        expected_shapes = [
            (batch_size, 96, 128, 128),
            (batch_size, 192, 64, 64),
            (batch_size, 384, 32, 32),
            (batch_size, 768, 16, 16),
        ]
        
        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            assert output.shape == expected_shape, \
                f"Stage {i+1}: Expected {expected_shape}, got {output.shape}"
        
        print("✓ Swin-Tiny forward pass successful")
        print(f"  Output shapes: {[out.shape for out in outputs]}")
    
    def test_different_input_sizes(self):
        """Test backbone with different input sizes."""
        backbone = SwinBackbone(num_channels=3, arch='swinb')
        backbone.eval()
        
        test_sizes = [(256, 256), (512, 512), (1024, 1024)]
        
        for h, w in test_sizes:
            input_tensor = torch.randn(1, 3, h, w)
            with torch.no_grad():
                outputs = backbone(input_tensor)
            
            # Verify output dimensions scale correctly
            assert outputs[0].shape[2] == h // 4, f"Height mismatch for input {h}x{w}"
            assert outputs[0].shape[3] == w // 4, f"Width mismatch for input {h}x{w}"
        
        print("✓ Different input sizes handled correctly")
    
    def test_invalid_architecture(self):
        """Test that invalid architecture raises error."""
        try:
            SwinBackbone(num_channels=3, arch='invalid')
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        print("✓ Invalid architecture correctly raises error")


class TestAggregationBackbone:
    """Test AggregationBackbone for multi-image input."""
    
    def test_aggregation_initialization(self):
        """Test AggregationBackbone initialization."""
        base_backbone = SwinBackbone(num_channels=3, arch='swinb')
        agg_backbone = AggregationBackbone(num_channels=3, backbone=base_backbone)
        
        assert agg_backbone is not None
        assert agg_backbone.image_channels == 3
        print("✓ AggregationBackbone initialization successful")
    
    def test_aggregation_forward_pass(self):
        """Test AggregationBackbone forward pass with multi-image input."""
        base_backbone = SwinBackbone(num_channels=3, arch='swinb')
        agg_backbone = AggregationBackbone(num_channels=3, backbone=base_backbone)
        agg_backbone.eval()
        
        # Multi-image input: (batch, num_images * channels, H, W)
        # Simulate 4 images
        batch_size = 2
        num_images = 4
        input_tensor = torch.randn(batch_size, num_images * 3, 512, 512)
        
        with torch.no_grad():
            outputs = agg_backbone(input_tensor)
        
        assert len(outputs) == 4
        # Aggregated features should have same spatial dimensions
        assert outputs[0].shape[0] == batch_size
        assert outputs[0].shape[2] == 128  # H/4
        assert outputs[0].shape[3] == 128  # W/4
        
        print("✓ AggregationBackbone forward pass successful")
        print(f"  Output shapes: {[out.shape for out in outputs]}")


class TestWeightLoading:
    """Test loading pretrained weights."""
    
    def test_weight_loading_structure(self):
        """Test that weight loading function exists and has correct signature."""
        # Just verify the function exists and can be called
        backbone = SwinBackbone(num_channels=3, arch='swinb')
        
        # Create a dummy checkpoint structure
        dummy_checkpoint = {
            'backbone.features.0.0.weight': torch.randn(128, 3, 4, 4),
            'backbone.features.0.0.bias': torch.randn(128),
        }
        
        # Save dummy checkpoint
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(dummy_checkpoint, f.name)
            temp_path = f.name
        
        try:
            # Test loading (will fail due to structure mismatch, but function should work)
            try:
                load_satlas_pretrained_weights(backbone, temp_path, multi_image=False)
            except (KeyError, RuntimeError):
                # Expected - dummy checkpoint doesn't match real structure
                pass
            
            print("✓ Weight loading function structure is correct")
        finally:
            os.unlink(temp_path)


def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 60)
    print("Testing SatlasPretrain Backbone Implementation")
    print("=" * 60)
    print()
    
    # Test SwinBackbone
    print("Testing SwinBackbone...")
    test_swin = TestSwinBackbone()
    test_swin.test_swinb_initialization()
    test_swin.test_swint_initialization()
    test_swin.test_swinb_forward_pass()
    test_swin.test_swint_forward_pass()
    test_swin.test_different_input_sizes()
    test_swin.test_invalid_architecture()
    print()
    
    # Test AggregationBackbone
    print("Testing AggregationBackbone...")
    test_agg = TestAggregationBackbone()
    test_agg.test_aggregation_initialization()
    test_agg.test_aggregation_forward_pass()
    print()
    
    # Test Weight Loading
    print("Testing Weight Loading...")
    test_weights = TestWeightLoading()
    test_weights.test_weight_loading_structure()
    print()
    
    print("=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

