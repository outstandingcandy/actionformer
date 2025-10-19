#!/usr/bin/env python3
"""
Video segment combination script
Randomly combines segments from the same original video to create new videos
"""

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile

def get_video_info(video_path: str) -> Tuple[float, str]:
    """Get video duration and frame rate using ffprobe"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
        str(video_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                duration = float(stream.get('duration', 0))
                fps = stream.get('r_frame_rate', '25/1')
                return duration, fps
                
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        pass
    
    return 0.0, '25/1'

def combine_segments(segment_files: List[str], output_path: str) -> bool:
    """Combine multiple video segments into one video using FFmpeg"""
    if not segment_files:
        return False
    
    # Create a temporary file list for FFmpeg concat
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for segment_file in segment_files:
            f.write(f"file '{segment_file}'\n")
    
    try:
        cmd = [
            'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
            '-c', 'copy', str(output_path), '-y'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error combining segments: {e}")
        print(f"FFmpeg stderr: {e.stderr}")
        return False
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(concat_file)
        except OSError:
            pass

def generate_random_combinations(video_id: str, segments: List[List[float]], 
                                labels: List[int], num_combinations: int = 10) -> List[Tuple[List[int], List[int]]]:
    """
    Generate random combinations of ALL segment indices and their corresponding labels
    Each combination includes all segments from the original video, but in different random orders
    
    Args:
        video_id: Original video identifier
        segments: List of [start_time, end_time] pairs
        labels: List of labels corresponding to segments
        num_combinations: Number of new combinations to generate
    
    Returns:
        List of tuples (segment_indices, segment_labels)
    """
    combinations = []
    num_segments = len(segments)
    
    # Create base list of all segment indices
    all_indices = list(range(num_segments))
    
    for i in range(num_combinations):
        # Create a copy of all indices and shuffle them randomly
        shuffled_indices = all_indices.copy()
        random.shuffle(shuffled_indices)
        
        # Get corresponding labels in the shuffled order
        shuffled_labels = [labels[idx] for idx in shuffled_indices]
        
        combinations.append((shuffled_indices, shuffled_labels))
    
    return combinations

def main():
    # Paths
    original_annotations_file = "/home/ubuntu/shuttle-sense/actionformer_release/data/actionformer_badminton_new/badminton_annotations.json"
    split_videos_dir = "/home/ubuntu/shuttle-sense/actionformer_release/output/split_videos"
    output_videos_dir = "/home/ubuntu/shuttle-sense/actionformer_release/output/combined_videos"
    output_annotations_file = "/home/ubuntu/shuttle-sense/actionformer_release/output/combined_badminton_annotations.json"
    
    # Create output directory
    Path(output_videos_dir).mkdir(parents=True, exist_ok=True)
    
    # Load original annotations
    try:
        with open(original_annotations_file, 'r') as f:
            original_annotations = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotations file not found: {original_annotations_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in annotations file: {e}")
        sys.exit(1)
    
    print(f"Loaded annotations for {len(original_annotations)} original videos")
    
    new_annotations = {}
    total_new_videos = 0
    
    # Process each original video
    for video_id, annotation in original_annotations.items():
        segments = annotation.get('segments', [])
        labels = annotation.get('labels', [])
        
        if not segments or not labels:
            print(f"Warning: No segments or labels found for video {video_id}")
            continue
        
        if len(segments) != len(labels):
            print(f"Warning: Segments and labels count mismatch for video {video_id}")
            continue
        
        print(f"Processing video {video_id} with {len(segments)} segments...")
        
        # Generate 10 random combinations for this video (all segments, different orders)
        combinations = generate_random_combinations(video_id, segments, labels, num_combinations=10)
        
        for combo_idx, (segment_indices, segment_labels) in enumerate(combinations):
            new_video_id = f"{video_id}_combo_{combo_idx:02d}"
            
            # Get paths to segment files
            segment_files = []
            for seg_idx in segment_indices:
                segment_file = Path(split_videos_dir) / f"{video_id}_segment_{seg_idx:03d}.mp4"
                if segment_file.exists():
                    segment_files.append(str(segment_file))
                else:
                    print(f"Warning: Segment file not found: {segment_file}")
            
            if not segment_files:
                print(f"Warning: No valid segment files for combination {new_video_id}")
                continue
            
            # Combine segments into new video
            output_video_path = Path(output_videos_dir) / f"{new_video_id}.mp4"
            
            print(f"  Creating {new_video_id} from {len(segment_files)} segments...")
            
            if combine_segments(segment_files, output_video_path):
                # Get video duration for the new combined video
                duration, fps = get_video_info(output_video_path)
                
                # Calculate new segment times (sequential concatenation)
                new_segments = []
                current_time = 0.0
                
                for seg_idx in segment_indices:
                    original_segment = segments[seg_idx]
                    segment_duration = original_segment[1] - original_segment[0]
                    new_segments.append([current_time, current_time + segment_duration])
                    current_time += segment_duration
                
                # Create annotation entry for new video
                new_annotations[new_video_id] = {
                    "duration": duration if duration > 0 else current_time,
                    "segments": new_segments,
                    "labels": segment_labels,
                    "original_video": video_id,
                    "segment_indices": segment_indices
                }
                
                total_new_videos += 1
                print(f"  ✓ Created {new_video_id}")
            else:
                print(f"  ✗ Failed to create {new_video_id}")
    
    # Save new annotations
    try:
        with open(output_annotations_file, 'w') as f:
            json.dump(new_annotations, f, indent=2)
        print(f"\nSaved new annotations to: {output_annotations_file}")
    except Exception as e:
        print(f"Error saving annotations: {e}")
        sys.exit(1)
    
    print(f"\nVideo combination completed!")
    print(f"Original videos: {len(original_annotations)}")
    print(f"New combined videos: {total_new_videos}")
    print(f"Multiplication factor: {total_new_videos / len(original_annotations):.1f}x")
    print(f"Combined videos saved to: {output_videos_dir}")

if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(42)
    main()