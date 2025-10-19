#!/usr/bin/env python3
"""
Generate splits.json and action_mapping.json for serve-only combined annotations
"""

import json
import sys
from pathlib import Path

def generate_serve_splits(serve_annotations: dict, original_splits: dict) -> dict:
    """
    Generate train/val splits for serve-only dataset based on original splits
    """
    serve_splits = {"train": [], "val": []}
    
    # Get all serve video IDs
    serve_video_ids = list(serve_annotations.keys())
    
    # Map serve videos to original videos and inherit the split
    for video_id in serve_video_ids:
        # Extract original video ID (remove _combo_XX suffix)
        original_video = serve_annotations[video_id].get('original_video', '')
        
        if original_video in original_splits['train']:
            serve_splits['train'].append(video_id)
        elif original_video in original_splits['val']:
            serve_splits['val'].append(video_id)
        else:
            # Default to train if not found
            serve_splits['train'].append(video_id)
    
    return serve_splits

def generate_serve_action_mapping() -> dict:
    """
    Generate action mapping for serve-only dataset (single class)
    """
    action_mapping = {
        "actions": [
            "发球"  # Single serve class
        ],
        "action_to_id": {
            "发球": 0
        },
        "hit_type_mapping": {
            "serve": "发球"
        },
        "original_serve_labels": {
            "3": "发球_bottom",
            "4": "发球_top"
        },
        "note": "Unified serve class combining original labels 3 (发球_bottom) and 4 (发球_top)"
    }
    
    return action_mapping

def main():
    # Paths
    serve_annotations_file = "/home/ubuntu/shuttle-sense/actionformer_release/output/serve_only_combined_annotations.json"
    original_splits_file = "/home/ubuntu/shuttle-sense/actionformer_release/data/actionformer_badminton_new/splits.json"
    
    output_splits_file = "/home/ubuntu/shuttle-sense/actionformer_release/output/serve_only_splits.json"
    output_action_mapping_file = "/home/ubuntu/shuttle-sense/actionformer_release/output/serve_only_action_mapping.json"
    
    # Load serve-only annotations
    try:
        with open(serve_annotations_file, 'r') as f:
            serve_annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Serve annotations file not found: {serve_annotations_file}")
        sys.exit(1)
    
    # Load original splits
    try:
        with open(original_splits_file, 'r') as f:
            original_splits = json.load(f)
    except FileNotFoundError:
        print(f"Error: Original splits file not found: {original_splits_file}")
        sys.exit(1)
    
    print(f"Loaded {len(serve_annotations)} serve-only videos")
    print(f"Original splits: {len(original_splits['train'])} train, {len(original_splits['val'])} val")
    
    # Generate serve splits
    serve_splits = generate_serve_splits(serve_annotations, original_splits)
    
    print(f"\\nGenerated serve splits:")
    print(f"  Train: {len(serve_splits['train'])} videos")
    print(f"  Val: {len(serve_splits['val'])} videos")
    
    # Generate serve action mapping
    serve_action_mapping = generate_serve_action_mapping()
    
    # Save files
    try:
        with open(output_splits_file, 'w', encoding='utf-8') as f:
            json.dump(serve_splits, f, indent=2, ensure_ascii=False)
        print(f"\\n✅ Saved serve splits to: {output_splits_file}")
        
        with open(output_action_mapping_file, 'w', encoding='utf-8') as f:
            json.dump(serve_action_mapping, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved serve action mapping to: {output_action_mapping_file}")
        
    except Exception as e:
        print(f"Error saving files: {e}")
        sys.exit(1)
    
    # Show sample splits
    print(f"\\n📊 Sample train videos:")
    for video in serve_splits['train'][:5]:
        original = serve_annotations[video]['original_video']
        print(f"  {video} (from {original})")
    
    print(f"\\n📊 Sample val videos:")
    for video in serve_splits['val'][:5]:
        original = serve_annotations[video]['original_video']
        print(f"  {video} (from {original})")
    
    print(f"\\n🎯 Action mapping:")
    print(f"  Actions: {serve_action_mapping['actions']}")
    print(f"  Total classes: {len(serve_action_mapping['actions'])}")
    
    print(f"\\n✅ Serve-only dataset files generated successfully!")

if __name__ == "__main__":
    main()