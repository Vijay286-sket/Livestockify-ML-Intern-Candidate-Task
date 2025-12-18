"""
Create a sample video from the dataset images for testing.
"""

import cv2
import numpy as np
from pathlib import Path
import os
import json

def create_sample_video():
    """Create a sample video from dataset images."""
    
    # Dataset path
    dataset_path = Path("C:/Users/vijay/.cache/kagglehub/datasets/hayriyigit/chicken-detection/versions/1")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = Path(root) / file
                image_files.append(image_path)
    
    # Sort files to maintain order
    image_files.sort()
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"Found {len(image_files)} image files")
    
    # Read first image to get dimensions
    first_image = cv2.imread(str(image_files[0]))
    if first_image is None:
        print("Could not read first image!")
        return
    
    height, width = first_image.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Create video writer
    output_path = "sample_chicken_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10  # 10 FPS
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write each frame multiple times to make video longer
    frames_per_image = 3  # Show each image for 3 frames
    
    for i, image_path in enumerate(image_files):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        
        # Resize to consistent dimensions if needed
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        # Write frame multiple times
        for _ in range(frames_per_image):
            out.write(img)
        
        print(f"Processed {i+1}/{len(image_files)}: {image_path.name}")
    
    # Release video writer
    out.release()
    
    print(f"Sample video created: {output_path}")
    print(f"Video duration: {len(image_files) * frames_per_image / fps:.2f} seconds")
    
    return output_path

if __name__ == "__main__":
    create_sample_video()
