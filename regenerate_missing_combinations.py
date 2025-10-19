#!/usr/bin/env python3
"""
Regenerate the missing 6 video combinations that failed during original processing
"""

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

def combine_segments(segment_files: List[str], output_path: str) -> bool:
    """Combine multiple video segments into one video using FFmpeg"""
    if not segment_files:
        return False
    
    # Create a temporary file list for FFmpeg concat
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        concat_file = f.name
        for segment_file in segment_files:
            # Use absolute paths to avoid issues
            abs_path = os.path.abspath(segment_file)
            f.write(f"file '{abs_path}'\n")
    
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

def get_video_info(video_path: str) -> float:
    """Get video duration using ffprobe"""
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
                return duration
                
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError):
        pass
    
    return 0.0

def generate_missing_combinations():
    """Generate the specific missing combinations using the same random seed"""
    
    # Set the same random seed as original script
    random.seed(42)
    
    # Load original annotations
    original_annotations_file = "data/actionformer_badminton_new/badminton_annotations.json"
    with open(original_annotations_file, 'r') as f:
        original_annotations = json.load(f)
    
    # Load existing combined annotations
    existing_annotations_file = "output/combined_badminton_annotations.json"
    with open(existing_annotations_file, 'r') as f:
        existing_annotations = json.load(f)
    
    # Define the missing combinations
    missing_combinations = [
        "0014_001_combo_00",
        "0014_001_combo_09", 
        "0014_003_combo_00",
        "0014_003_combo_05",
        "0014_003_combo_06", 
        "0014_003_combo_09"
    ]
    
    split_videos_dir = "output/split_videos"
    output_videos_dir = "output/combined_videos"
    
    successful_recreations = []
    
    for missing_combo in missing_combinations:
        # Parse the combo name
        parts = missing_combo.split('_combo_')
        video_id = parts[0]
        combo_idx = int(parts[1])
        
        print(f"\nProcessing missing combination: {missing_combo}")
        
        # Get original video data
        if video_id not in original_annotations:
            print(f"  ✗ Original video {video_id} not found in annotations")
            continue
            
        segments = original_annotations[video_id]['segments']
        labels = original_annotations[video_id]['labels']
        
        # Simulate the random generation process to get the exact same indices
        # We need to generate all combinations up to the one we want
        num_segments = len(segments)
        all_indices = list(range(num_segments))
        
        target_indices = None
        target_labels = None
        
        for i in range(10):  # Generate all 10 combinations
            shuffled_indices = all_indices.copy()
            random.shuffle(shuffled_indices)
            shuffled_labels = [labels[idx] for idx in shuffled_indices]
            
            if i == combo_idx:
                target_indices = shuffled_indices
                target_labels = shuffled_labels
                break
        
        if target_indices is None:
            print(f"  ✗ Could not generate indices for combo {combo_idx}")
            continue
        
        print(f"  Target indices: {target_indices}")
        
        # Get segment file paths
        segment_files = []
        missing_segments = []
        
        for seg_idx in target_indices:
            segment_file = Path(split_videos_dir) / f"{video_id}_segment_{seg_idx:03d}.mp4"
            if segment_file.exists():
                segment_files.append(str(segment_file))
            else:
                missing_segments.append(str(segment_file))
        
        if missing_segments:
            print(f"  ✗ Missing segment files: {missing_segments}")
            continue
        
        # Combine segments
        output_video_path = Path(output_videos_dir) / f"{missing_combo}.mp4"
        
        print(f"  Combining {len(segment_files)} segments...")
        
        if combine_segments(segment_files, output_video_path):
            # Get video duration
            duration = get_video_info(output_video_path)
            
            # Calculate new segment times
            new_segments = []
            current_time = 0.0
            
            for seg_idx in target_indices:
                original_segment = segments[seg_idx]
                segment_duration = original_segment[1] - original_segment[0]
                new_segments.append([current_time, current_time + segment_duration])
                current_time += segment_duration
            
            # Create annotation entry
            new_annotation = {
                "duration": duration if duration > 0 else current_time,
                "segments": new_segments,
                "labels": target_labels,
                "original_video": video_id,
                "segment_indices": target_indices
            }
            
            # Add to existing annotations
            existing_annotations[missing_combo] = new_annotation
            successful_recreations.append(missing_combo)
            
            print(f"  ✓ Successfully created {missing_combo}")
        else:
            print(f"  ✗ Failed to create {missing_combo}")
    
    # Save updated annotations if any combinations were successful
    if successful_recreations:
        with open(existing_annotations_file, 'w') as f:
            json.dump(existing_annotations, f, indent=2)
        
        print(f"\n✅ Successfully regenerated {len(successful_recreations)} missing combinations:")
        for combo in successful_recreations:
            print(f"   {combo}")
        
        print(f"\nUpdated annotations saved to: {existing_annotations_file}")
        print(f"Total combined videos: {len(existing_annotations)}")
    else:
        print(f"\n❌ No combinations were successfully regenerated")

def main():
    print("🔄 Regenerating missing video combinations...")
    
    # Clean up any test files first
    test_files = ["test_0014_001_combo_00.mp4"]
    for test_file in test_files:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"Cleaned up test file: {test_file}")
    
    generate_missing_combinations()
    
    print(f"\n🎯 Missing combination regeneration completed!")

if __name__ == "__main__":
    main()