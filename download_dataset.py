"""
Script to download and explore the chicken detection dataset.
"""

import kagglehub
import os
from pathlib import Path
import cv2
import json

def download_dataset():
    """Download the chicken detection dataset."""
    print("Downloading chicken detection dataset...")
    
    # Download latest version
    path = kagglehub.dataset_download("hayriyigit/chicken-detection")
    
    print(f"Path to dataset files: {path}")
    
    # Explore dataset structure
    explore_dataset(path)
    
    return path

def explore_dataset(dataset_path):
    """Explore and print dataset structure."""
    print("\n=== Dataset Structure ===")
    
    dataset_path = Path(dataset_path)
    
    # List all files and directories
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(str(dataset_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Limit to first 5 files to avoid clutter
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")

def find_sample_videos(dataset_path):
    """Find sample video files in the dataset."""
    print("\n=== Looking for video files ===")
    
    dataset_path = Path(dataset_path)
    video_files = []
    
    # Common video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = Path(root) / file
                video_files.append(video_path)
                print(f"Found video: {video_path}")
    
    return video_files

def analyze_video_sample(video_path):
    """Analyze a sample video to understand its properties."""
    print(f"\n=== Analyzing video: {video_path} ===")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"FPS: {fps}")
    print(f"Frame count: {frame_count}")
    print(f"Resolution: {width}x{height}")
    print(f"Duration: {duration:.2f} seconds")
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }

if __name__ == "__main__":
    # Download and explore dataset
    dataset_path = download_dataset()
    
    # Find sample videos
    video_files = find_sample_videos(dataset_path)
    
    # Analyze first video found
    if video_files:
        video_info = analyze_video_sample(video_files[0])
        
        # Save dataset info
        info = {
            'dataset_path': str(dataset_path),
            'video_files': [str(v) for v in video_files],
            'video_info': video_info
        }
        
        with open('dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\nDataset info saved to dataset_info.json")
    else:
        print("No video files found in dataset")
