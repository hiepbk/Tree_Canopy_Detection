"""
Visualization tool for tree canopy dataset.
Displays images with segmentation ground truth annotations.
"""

import argparse
import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from pathlib import Path


def load_coco_annotations(ann_file):
    """Load COCO format annotations."""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    return data


def get_image_annotations(coco_data, image_id):
    """Get all annotations for a specific image."""
    annotations = []
    for ann in coco_data.get('annotations', []):
        if ann['image_id'] == image_id:
            annotations.append(ann)
    return annotations


def polygon_to_coords(segmentation):
    """
    Convert COCO segmentation format to coordinate pairs.
    
    COCO format: [[x1, y1, x2, y2, ...]] or [x1, y1, x2, y2, ...]
    Returns: list of (x, y) tuples
    """
    if not segmentation:
        return []
    
    # Handle list of lists format
    if isinstance(segmentation[0], list):
        coords = segmentation[0]
    else:
        coords = segmentation
    
    # Convert to (x, y) pairs
    points = []
    for i in range(0, len(coords), 2):
        if i + 1 < len(coords):
            points.append((coords[i], coords[i + 1]))
    
    return points


def visualize_image_with_annotations(image_path, coco_data, image_id):
    """
    Visualize an image with its segmentation annotations.
    
    Args:
        image_path: Path to the image file
        coco_data: COCO annotation data dictionary
        image_id: ID of the image to visualize
    """
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Read image (handle TIFF format)
    img = cv2.imread(image_path)
    if img is None:
        # Try with different backend
        from PIL import Image
        img = np.array(Image.open(image_path))
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]  # Remove alpha channel if present
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get image info
    image_info = None
    for img_info in coco_data.get('images', []):
        if img_info['id'] == image_id:
            image_info = img_info
            break
    
    if image_info is None:
        print(f"Error: Image ID {image_id} not found in annotations")
        return
    
    # Get annotations for this image
    annotations = get_image_annotations(coco_data, image_id)
    
    # Get category mapping
    category_map = {cat['id']: cat for cat in coco_data.get('categories', [])}
    
    # Color mapping for different classes
    class_colors = {
        1: (0, 255, 0),      # Green for individual_tree
        2: (255, 0, 0),      # Red for group_of_trees
    }
    class_names = {
        1: 'individual_tree',
        2: 'group_of_trees',
    }
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img_rgb)
    ax.set_title(f"Image: {image_info['file_name']}\n"
                 f"Size: {image_info['width']}x{image_info['height']} | "
                 f"Annotations: {len(annotations)}", 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Draw annotations
    patches = []
    colors_list = []
    legend_elements = []
    drawn_classes = set()
    
    for ann in annotations:
        category_id = ann['category_id']
        category_name = category_map.get(category_id, {}).get('name', f'class_{category_id}')
        
        # Get color for this class
        color = class_colors.get(category_id, (255, 255, 0))  # Yellow as default
        color_normalized = tuple(c / 255.0 for c in color)
        
        # Convert segmentation to polygon
        segmentation = ann.get('segmentation', [])
        if not segmentation:
            continue
        
        # Handle multiple polygons (COCO can have multiple polygons per annotation)
        if isinstance(segmentation[0], list):
            # List of polygons
            for poly_seg in segmentation:
                coords = polygon_to_coords(poly_seg)
                if len(coords) >= 3:  # Need at least 3 points for a polygon
                    polygon = Polygon(coords, closed=True, fill=False, 
                                    edgecolor=color_normalized, linewidth=2)
                    patches.append(polygon)
                    colors_list.append(color_normalized)
        else:
            # Single polygon
            coords = polygon_to_coords(segmentation)
            if len(coords) >= 3:
                polygon = Polygon(coords, closed=True, fill=False,
                                edgecolor=color_normalized, linewidth=2)
                patches.append(polygon)
                colors_list.append(color_normalized)
        
        # Add to legend if not already added
        if category_id not in drawn_classes:
            from matplotlib.patches import Patch
            legend_elements.append(
                Patch(facecolor=color_normalized, edgecolor=color_normalized, 
                     label=f'{category_name} (ID: {category_id})')
            )
            drawn_classes.add(category_id)
    
    # Add polygons to plot
    if patches:
        p = PatchCollection(patches, match_original=True, alpha=0.7)
        ax.add_collection(p)
    
    # Add legend
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add statistics text
    stats_text = f"Total instances: {len(annotations)}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()  # Blocks until window is closed
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize tree canopy dataset images with ground truth annotations'
    )
    parser.add_argument(
        '--ann-file',
        type=str,
        default='data/Tree_Canopy_Data/annotations/instances_train.json',
        help='Path to COCO annotation file'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='data/Tree_Canopy_Data/raw_data/trainval_images',
        help='Directory containing images'
    )
    parser.add_argument(
        '--image-id',
        type=int,
        default=None,
        help='Specific image ID to visualize (if not provided, shows first image)'
    )
    parser.add_argument(
        '--image-name',
        type=str,
        default=None,
        help='Specific image filename to visualize'
    )
    
    args = parser.parse_args()
    
    # Load annotations
    print(f"Loading annotations from {args.ann_file}...")
    coco_data = load_coco_annotations(args.ann_file)
    print(f"Loaded {len(coco_data.get('images', []))} images")
    print(f"Loaded {len(coco_data.get('annotations', []))} annotations")
    print(f"Categories: {[cat['name'] for cat in coco_data.get('categories', [])]}")
    
    # Determine which images to visualize
    images_to_show = []
    
    if args.image_name:
        # Find image by filename
        for img_info in coco_data.get('images', []):
            if img_info['file_name'] == args.image_name:
                images_to_show.append(img_info)
                break
        if not images_to_show:
            print(f"Error: Image '{args.image_name}' not found in annotations")
            return
    elif args.image_id:
        # Find image by ID
        for img_info in coco_data.get('images', []):
            if img_info['id'] == args.image_id:
                images_to_show.append(img_info)
                break
        if not images_to_show:
            print(f"Error: Image ID {args.image_id} not found in annotations")
            return
    else:
        # Show all images
        images_to_show = coco_data.get('images', [])
        if len(images_to_show) == 0:
            print("Error: No images found in annotations")
            return
    
    # Visualize each image one by one
    print(f"\nShowing {len(images_to_show)} image(s). Close each window to see the next one.\n")
    
    for idx, img_info in enumerate(images_to_show, 1):
        image_path = os.path.join(args.image_dir, img_info['file_name'])
        
        print(f"[{idx}/{len(images_to_show)}] Visualizing image ID {img_info['id']}: {img_info['file_name']}")
        
        visualize_image_with_annotations(
            image_path, 
            coco_data, 
            img_info['id']
        )
    
    print("\n✓ Finished visualizing all images!")


if __name__ == '__main__':
    main()

