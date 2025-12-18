#!/usr/bin/env python3
"""
Test script for the Bird Counting and Weight Estimation API
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8001"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing health endpoint: {e}")
        return False

def test_analyze_video():
    """Test the video analysis endpoint"""
    print("\nTesting video analysis endpoint...")
    
    # Prepare the video file
    video_path = "sample_chicken_video.mp4"
    
    try:
        with open(video_path, 'rb') as video_file:
            files = {
                'video': ('sample_chicken_video.mp4', video_file, 'video/mp4')
            }
            data = {
                'fps_sample': 2,
                'conf_thresh': 0.5,
                'iou_thresh': 0.45
            }
            
            print("Sending request to analyze video...")
            print("This may take a few minutes...")
            
            response = requests.post(
                f"{BASE_URL}/analyze_video",
                files=files,
                data=data,
                timeout=300  # 5 minute timeout
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("Analysis completed successfully!")
                print(f"Number of count entries: {len(result.get('counts', []))}")
                print(f"Number of track samples: {len(result.get('tracks_sample', []))}")
                print(f"Weight estimates: {result.get('weight_estimates', {})}")
                print(f"Artifacts: {result.get('artifacts', {})}")
                
                # Save the full response to a file
                with open('api_response_sample.json', 'w') as f:
                    json.dump(result, f, indent=2)
                print("Full response saved to 'api_response_sample.json'")
                
                return True
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except FileNotFoundError:
        print(f"Error: Video file '{video_path}' not found")
        return False
    except Exception as e:
        print(f"Error testing video analysis: {e}")
        return False

def main():
    """Main test function"""
    print("=== Bird Counting and Weight Estimation API Test ===")
    
    # Test health endpoint
    health_ok = test_health()
    
    if not health_ok:
        print("Health check failed. Make sure the API server is running.")
        return
    
    # Test video analysis
    analysis_ok = test_analyze_video()
    
    if analysis_ok:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed.")

if __name__ == "__main__":
    main()