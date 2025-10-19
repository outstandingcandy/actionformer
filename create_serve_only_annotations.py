#!/usr/bin/env python3
"""
Create serve-only annotations while keeping the original combined videos
Only modify the annotation file to keep serve segments (labels 3 and 4)
"""

import json
import sys

def filter_serve_only_annotations(annotations: dict, serve_labels: list = [3, 4]) -> dict:
    """
    Filter annotations to keep only serve segments while preserving video files
    
    Args:
        annotations: Original combined annotations dict
        serve_labels: List of label IDs that represent serves (default: [3, 4])
    
    Returns:
        Filtered annotations dict with only serve segments
    """
    serve_only_annotations = {}
    
    for video_id, annotation in annotations.items():
        segments = annotation.get('segments', [])
        labels = annotation.get('labels', [])
        duration = annotation.get('duration', 0)
        original_video = annotation.get('original_video', video_id)
        segment_indices = annotation.get('segment_indices', [])
        
        # Find serve segments and their indices
        serve_segments = []
        serve_segment_labels = []
        serve_segment_indices = []
        
        for i, (seg, label) in enumerate(zip(segments, labels)):
            if label in serve_labels:
                serve_segments.append(seg)
                # Convert serve labels to single class (0)
                serve_segment_labels.append(0)
                # Keep track of original segment indices if available
                if i < len(segment_indices):
                    serve_segment_indices.append(segment_indices[i])
        
        # Only include videos that have serve segments
        if serve_segments:
            serve_only_annotations[video_id] = {
                "duration": duration,
                "segments": serve_segments,
                "labels": serve_segment_labels,
                "original_video": original_video,
                "segment_indices": serve_segment_indices if serve_segment_indices else None,
                "original_total_segments": len(segments),
                "serve_segments_count": len(serve_segments),
                "original_serve_labels": [labels[i] for i, label in enumerate(labels) if label in serve_labels]
            }
    
    return serve_only_annotations

def main():
    # Paths
    input_annotations_file = "/home/ubuntu/shuttle-sense/actionformer_release/output/combined_badminton_annotations.json"
    output_annotations_file = "/home/ubuntu/shuttle-sense/actionformer_release/output/serve_only_combined_annotations.json"
    
    # Load combined annotations
    try:
        with open(input_annotations_file, 'r') as f:
            combined_annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Combined annotations file not found: {input_annotations_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in annotations file: {e}")
        sys.exit(1)
    
    print(f"Loaded {len(combined_annotations)} combined videos")
    
    # Serve labels based on analysis (labels 3 and 4 appear at video start)
    serve_labels = [3, 4]
    print(f"Filtering for serve labels: {serve_labels}")
    
    # Filter annotations for serve-only segments
    serve_only_annotations = filter_serve_only_annotations(combined_annotations, serve_labels)
    
    print(f"\\nFiltered to {len(serve_only_annotations)} videos with serve segments")
    
    # Calculate statistics
    total_original_segments = sum(ann.get('original_total_segments', 0) for ann in serve_only_annotations.values())
    total_serve_segments = sum(ann.get('serve_segments_count', 0) for ann in serve_only_annotations.values())
    
    print(f"Total original segments: {total_original_segments}")
    print(f"Total serve segments: {total_serve_segments}")
    print(f"Serve ratio: {total_serve_segments/total_original_segments*100:.1f}%")
    
    # Show examples of filtered data
    print(f"\\nSample filtered annotations:")
    for i, (video_id, annotation) in enumerate(list(serve_only_annotations.items())[:3]):
        original_labels = annotation.get('original_serve_labels', [])
        serve_count = annotation.get('serve_segments_count', 0)
        total_count = annotation.get('original_total_segments', 0)
        print(f"  {video_id}: {serve_count}/{total_count} segments kept, original serve labels: {original_labels}")
    
    # Save serve-only annotations
    try:
        with open(output_annotations_file, 'w') as f:
            json.dump(serve_only_annotations, f, indent=2)
        print(f"\\nSaved serve-only annotations to: {output_annotations_file}")
    except Exception as e:
        print(f"Error saving annotations: {e}")
        sys.exit(1)
    
    print(f"\\n✅ Serve-only annotation filtering completed!")
    print(f"📁 Video files: Use existing combined_videos/ directory")
    print(f"📄 Annotations: {output_annotations_file}")
    print(f"🎯 Videos with serves: {len(serve_only_annotations)}")
    print(f"📊 Total serve segments: {total_serve_segments}")
    print(f"🏷️  All serve labels converted to: 0 (single class)")

if __name__ == "__main__":
    main()