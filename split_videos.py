#!/usr/bin/env python3
"""
Video splitting script based on badminton annotations
"""

import json
import os
import subprocess
import sys
from pathlib import Path

def split_video(video_path, output_dir, segments, video_id):
    """
    Split a video into segments using FFmpeg
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save split videos
        segments: List of [start_time, end_time] pairs
        video_id: Video identifier for output naming
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        print(f"Warning: Video file {video_path} does not exist")
        return
    
    print(f"Splitting video {video_id} into {len(segments)} segments...")
    
    for i, (start_time, end_time) in enumerate(segments):
        duration = end_time - start_time
        output_file = output_dir / f"{video_id}_segment_{i:03d}.mp4"
        
        # FFmpeg command to extract segment
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c', 'copy',  # Copy streams without re-encoding for speed
            '-avoid_negative_ts', 'make_zero',
            str(output_file),
            '-y'  # Overwrite output files
        ]
        
        try:
            # Run FFmpeg command
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  check=True)
            print(f"  Created: {output_file.name}")
            
        except subprocess.CalledProcessError as e:
            print(f"  Error creating {output_file.name}: {e}")
            print(f"  FFmpeg stderr: {e.stderr}")
        except FileNotFoundError:
            print("Error: FFmpeg not found. Please install FFmpeg.")
            sys.exit(1)

def main():
    # Paths
    annotations_file = "/home/ubuntu/shuttle-sense/actionformer_release/data/actionformer_badminton_new/badminton_annotations.json"
    videos_dir = "/data/badmiton_data/videos"
    output_base_dir = "/home/ubuntu/shuttle-sense/actionformer_release/output/split_videos"
    
    # Load annotations
    try:
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotations file not found: {annotations_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in annotations file: {e}")
        sys.exit(1)
    
    print(f"Loaded annotations for {len(annotations)} videos")
    
    # Process each video
    for video_id, annotation in annotations.items():
        video_file = Path(videos_dir) / f"{video_id}.mp4"
        segments = annotation.get('segments', [])
        
        if not segments:
            print(f"Warning: No segments found for video {video_id}")
            continue
            
        split_video(video_file, output_base_dir, segments, video_id)
    
    print(f"\nVideo splitting completed!")
    print(f"Split videos saved to: {output_base_dir}")

if __name__ == "__main__":
    main()