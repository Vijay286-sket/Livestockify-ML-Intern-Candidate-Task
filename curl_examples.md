# API Testing Examples

## Health Check
```bash
curl http://localhost:8001/health
```

## Analyze Video (Synchronous)
```bash
curl -X POST "http://localhost:8001/analyze_video" \
  -H "accept: application/json" \
  -F "video=@sample_chicken_video.mp4" \
  -F "fps_sample=2" \
  -F "conf_thresh=0.5" \
  -F "iou_thresh=0.45"
```

## PowerShell Example
```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8001/health" -Method Get

# Video analysis (simplified)
$uri = "http://localhost:8001/analyze_video"
$videoPath = "sample_chicken_video.mp4"
$form = @{
    video = Get-Item $videoPath
    fps_sample = 2
    conf_thresh = 0.5
    iou_thresh = 0.45
}
Invoke-RestMethod -Uri $uri -Method Post -Form $form
```

## Python Example
```python
import requests

# Health check
response = requests.get("http://localhost:8001/health")
print(response.json())

# Video analysis
with open("sample_chicken_video.mp4", "rb") as video_file:
    files = {"video": ("sample_chicken_video.mp4", video_file, "video/mp4")}
    data = {
        "fps_sample": 2,
        "conf_thresh": 0.5,
        "iou_thresh": 0.45
    }
    response = requests.post("http://localhost:8001/analyze_video", files=files, data=data)
    result = response.json()
    print(result)
```

## Expected Response Structure
```json
{
  "counts": [
    {"timestamp": 0.0, "count": 3},
    {"timestamp": 0.5, "count": 4}
  ],
  "tracks_sample": [
    {
      "track_id": 1,
      "bbox": [100, 120, 200, 260],
      "confidence": 0.85,
      "weight": 1500.0,
      "weight_confidence": 0.75
    }
  ],
  "weight_estimates": {
    "total_weight": 6000.0,
    "average_weight": 1500.0,
    "unit": "grams (estimated)"
  },
  "confidence": {
    "average_confidence": 0.80,
    "min_confidence": 0.65,
    "max_confidence": 0.95
  },
  "artifacts": {
    "annotated_video": "outputs/sample_chicken_video_annotated.mp4",
    "results_json": "outputs/analysis_results.json"
  },
  "metadata": {
    "fps_sample": 2,
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45
  }
}
```