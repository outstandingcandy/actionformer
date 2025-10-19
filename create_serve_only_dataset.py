#!/usr/bin/env python3
"""
Create serve-only badminton annotations
Based on analysis, labels 3 and 4 appear to be serve actions
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile

def filter_serve_segments(annotations: Dict, serve_labels: List[int] = [3, 4]) -> Dict:
    """
    Filter annotations to keep only serve segments
    
    Args:
        annotations: Original annotations dict
        serve_labels: List of label IDs that represent serves (default: [3, 4])
    
    Returns:
        Filtered annotations dict with only serve segments
    """
    serve_annotations = {}
    
    for video_id, annotation in annotations.items():
        segments = annotation.get('segments', [])
        labels = annotation.get('labels', [])
        duration = annotation.get('duration', 0)
        
        # Find serve segments
        serve_segments = []
        serve_segment_labels = []
        
        for seg, label in zip(segments, labels):
            if label in serve_labels:
                serve_segments.append(seg)
                # Convert all serve labels to a single class (0 for binary classification)
                serve_segment_labels.append(0)
        
        # Only include videos that have serve segments
        if serve_segments:
            serve_annotations[video_id] = {
                "duration": duration,
                "segments": serve_segments,
                "labels": serve_segment_labels,
                "original_video": video_id,
                "original_segments_count": len(segments),
                "serve_segments_count": len(serve_segments)
            }
    
    return serve_annotations

def create_serve_video_segments(serve_annotations: Dict, 
                               split_videos_dir: str, 
                               output_dir: str) -> None:
    """
    Create video files containing only serve segments for each video
    
    Args:
        serve_annotations: Filtered serve-only annotations
        split_videos_dir: Directory containing split video segments
        output_dir: Directory to save serve-only videos
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for video_id, annotation in serve_annotations.items():
        segments = annotation.get('segments', [])
        
        if not segments:
            continue
        
        print(f"Creating serve-only video for {video_id} with {len(segments)} serve segments...")
        
        # Find corresponding segment files
        segment_files = []
        for i, segment in enumerate(segments):
            # Find the original segment index that corresponds to this serve segment
            # We need to map back to the original segment files
            
            # For combined videos, we need to extract from the combined video
            combined_video_path = None
            
            # Check if this is a combined video
            if '_combo_' in video_id:
                combined_video_path = Path("/home/ubuntu/shuttle-sense/actionformer_release/output/combined_videos") / f"{video_id}.mp4"
                if combined_video_path.exists():
                    # Extract serve segments directly from combined video
                    extract_segments_from_combined_video(
                        combined_video_path, 
                        segments, 
                        output_dir / f"{video_id}_serve_only.mp4"
                    )
                    continue
            
            # For original videos, find segments from split directory
            base_video_id = video_id.split('_combo_')[0] if '_combo_' in video_id else video_id
            
            # We need to find which original segments correspond to serve labels
            # This requires referencing the original annotations
            print(f"  Warning: Unable to directly map segments for {video_id}")
    
    print(f"Serve-only videos saved to: {output_dir}")

def extract_segments_from_combined_video(video_path: Path, 
                                       segments: List, 
                                       output_path: Path) -> bool:
    """
    Extract serve segments from a combined video and concatenate them
    """
    try:
        temp_files = []
        
        # Extract each serve segment as a temporary file
        for i, segment in enumerate(segments):
            start_time = segment[0]
            end_time = segment[1]
            duration = end_time - start_time
            
            temp_file = output_path.parent / f"temp_serve_{i}.mp4"
            temp_files.append(temp_file)
            
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',
                str(temp_file),
                '-y'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Concatenate all serve segments
        if len(temp_files) > 1:
            # Create concat file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                concat_file = f.name
                for temp_file in temp_files:
                    f.write(f"file '{temp_file}'\\n")
            
            cmd = [
                'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
                '-c', 'copy', str(output_path), '-y'
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            os.unlink(concat_file)
        
        elif len(temp_files) == 1:
            # Just rename the single file
            temp_files[0].rename(output_path)
            temp_files = []  # Prevent cleanup
        
        # Clean up temporary files
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  Error extracting segments: {e}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    # Paths
    original_annotations_file = "/home/ubuntu/shuttle-sense/actionformer_release/data/actionformer_badminton_new/badminton_annotations.json"
    combined_annotations_file = "/home/ubuntu/shuttle-sense/actionformer_release/output/combined_badminton_annotations.json"
    output_annotations_file = "/home/ubuntu/shuttle-sense/actionformer_release/output/serve_only_badminton_annotations.json"
    output_videos_dir = "/home/ubuntu/shuttle-sense/actionformer_release/output/serve_only_videos"
    
    # Load original annotations to understand serve labels
    try:
        with open(original_annotations_file, 'r') as f:
            original_annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Original annotations file not found: {original_annotations_file}")
        sys.exit(1)
    
    # Load combined annotations
    try:
        with open(combined_annotations_file, 'r') as f:
            combined_annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Combined annotations file not found: {combined_annotations_file}")
        sys.exit(1)
    
    print(f"Loaded {len(original_annotations)} original videos")
    print(f"Loaded {len(combined_annotations)} combined videos")
    
    # Serve labels based on analysis (labels 3 and 4 appear at video start)
    serve_labels = [3, 4]
    print(f"Using serve labels: {serve_labels}")
    
    # Filter combined annotations for serve-only segments
    serve_annotations = filter_serve_segments(combined_annotations, serve_labels)
    
    print(f"\\nFiltered to {len(serve_annotations)} videos with serve segments")
    
    # Calculate statistics
    total_original_segments = sum(ann.get('original_segments_count', 0) for ann in serve_annotations.values())
    total_serve_segments = sum(ann.get('serve_segments_count', 0) for ann in serve_annotations.values())
    
    print(f"Total original segments: {total_original_segments}")
    print(f"Total serve segments: {total_serve_segments}")
    print(f"Serve ratio: {total_serve_segments/total_original_segments*100:.1f}%")
    
    # Create serve-only video files
    create_serve_video_segments(serve_annotations, "", output_videos_dir)
    
    # Save serve-only annotations
    try:
        with open(output_annotations_file, 'w') as f:
            json.dump(serve_annotations, f, indent=2)
        print(f"\\nSaved serve-only annotations to: {output_annotations_file}")
    except Exception as e:
        print(f"Error saving annotations: {e}")
        sys.exit(1)
    
    print(f"\\nServe-only dataset creation completed!")
    print(f"Videos: {len(serve_annotations)}")
    print(f"Total serve segments: {total_serve_segments}")
    print(f"Videos saved to: {output_videos_dir}")

if __name__ == "__main__":
    main()