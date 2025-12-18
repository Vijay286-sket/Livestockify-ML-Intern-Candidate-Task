"""
Main entry point for the Bird Counting and Weight Estimation system.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.video_processor import VideoProcessor
from src.config import Config

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Bird Counting and Weight Estimation from CCTV Video"
    )
    
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Config.OUTPUT_DIR),
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=Config.CONFIDENCE_THRESHOLD,
        help="Detection confidence threshold"
    )
    
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=Config.IOU_THRESHOLD,
        help="IoU threshold for detection"
    )
    
    parser.add_argument(
        "--fps-sample",
        type=int,
        default=Config.DEFAULT_FPS_SAMPLE,
        help="Frame sampling rate (process every Nth frame)"
    )
    
    args = parser.parse_args()
    
    # Validate input video
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Starting video analysis...")
    print(f"Input: {video_path}")
    print(f"Output: {output_dir}")
    print(f"Parameters: conf_thresh={args.conf_thresh}, iou_thresh={args.iou_thresh}, fps_sample={args.fps_sample}")
    
    try:
        # Initialize processor
        processor = VideoProcessor(
            conf_thresh=args.conf_thresh,
            iou_thresh=args.iou_thresh,
            fps_sample=args.fps_sample
        )
        
        # Process video
        results = processor.process_video(video_path, output_dir)
        
        # Print summary
        print("\n=== Processing Complete ===")
        print(f"Bird Count Statistics:")
        print(f"  Average: {results['statistics']['bird_count']['average']:.1f}")
        print(f"  Maximum: {results['statistics']['bird_count']['maximum']}")
        print(f"  Minimum: {results['statistics']['bird_count']['minimum']}")
        
        print(f"\nWeight Estimation:")
        print(f"  Average Total Weight: {results['statistics']['weight']['average_total_weight']:.0f}g")
        print(f"  Maximum Total Weight: {results['statistics']['weight']['maximum_total_weight']:.0f}g")
        
        print(f"\nGenerated Files:")
        for key, path in results['artifacts'].items():
            print(f"  {key}: {path}")
        
        print(f"\nNotes:")
        for note in results['notes']:
            print(f"  - {note}")
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
