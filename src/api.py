"""
FastAPI service for bird counting and weight estimation from video.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import uuid
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Support both "run as script" (python src/api.py) and "package" imports
# (python -m uvicorn src.api:app) by trying local and package-relative.
try:  # script-style
    from video_processor import VideoProcessor
    from config import Config
except ModuleNotFoundError:  # package-style (src.api)
    from .video_processor import VideoProcessor  # type: ignore
    from .config import Config  # type: ignore

# Initialize FastAPI app
app = FastAPI(
    title="Bird Counting and Weight Estimation API",
    description="API for counting birds and estimating weights from CCTV video footage",
    version="1.0.0"
)

# Create necessary directories
Config.create_directories()

# Global storage for processing tasks
processing_tasks = {}

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    version: str

class VideoAnalysisRequest(BaseModel):
    """Video analysis request parameters."""
    fps_sample: Optional[int] = Config.DEFAULT_FPS_SAMPLE
    conf_thresh: Optional[float] = Config.CONFIDENCE_THRESHOLD
    iou_thresh: Optional[float] = Config.IOU_THRESHOLD

class VideoAnalysisResponse(BaseModel):
    """Video analysis response model."""
    task_id: str
    status: str
    message: str

class AnalysisResult(BaseModel):
    """Analysis result model."""
    counts: list
    tracks_sample: list
    weight_estimates: dict
    confidence: dict
    artifacts: dict
    metadata: dict

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="OK",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/analyze_video_async", response_model=VideoAnalysisResponse)
async def analyze_video_async(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    fps_sample: Optional[int] = Config.DEFAULT_FPS_SAMPLE,
    conf_thresh: Optional[float] = Config.CONFIDENCE_THRESHOLD,
    iou_thresh: Optional[float] = Config.IOU_THRESHOLD
):
    """
    Analyze video for bird counting and weight estimation.
    
    Args:
        video: Video file to analyze
        fps_sample: Frame sampling rate (process every Nth frame)
        conf_thresh: Detection confidence threshold
        iou_thresh: IoU threshold for detection
        
    Returns:
        Task ID for tracking analysis progress
    """
    
    # Validate file type
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    file_ext = Path(video.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Check file size
    file_size = 0
    video.file.seek(0, 2)  # Seek to end
    file_size = video.file.tell()
    video.file.seek(0)  # Reset to beginning
    
    max_size_bytes = Config.MAX_VIDEO_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {Config.MAX_VIDEO_SIZE_MB}MB"
        )
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Create task directory
    task_dir = Config.UPLOAD_DIR / task_id
    task_dir.mkdir(exist_ok=True)
    
    # Save uploaded video
    video_path = task_dir / video.filename
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Initialize task status
        processing_tasks[task_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Video uploaded, processing queued",
            "created_at": datetime.now().isoformat(),
            "video_path": str(video_path),
            "output_dir": str(task_dir),
            "parameters": {
                "fps_sample": fps_sample,
                "conf_thresh": conf_thresh,
                "iou_thresh": iou_thresh
            }
        }
        
        # Add background processing task
        background_tasks.add_task(
            process_video_task,
            task_id,
            str(video_path),
            str(task_dir),
            fps_sample,
            conf_thresh,
            iou_thresh
        )
        
        return VideoAnalysisResponse(
            task_id=task_id,
            status="queued",
            message="Video uploaded successfully. Processing started."
        )
        
    except Exception as e:
        # Cleanup on error
        if video_path.exists():
            video_path.unlink()
        if task_dir.exists():
            shutil.rmtree(task_dir)
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save video: {str(e)}"
        )


@app.post("/analyze_video", response_model=AnalysisResult)
async def analyze_video(
    video: UploadFile = File(...),
    fps_sample: Optional[int] = Config.DEFAULT_FPS_SAMPLE,
    conf_thresh: Optional[float] = Config.CONFIDENCE_THRESHOLD,
    iou_thresh: Optional[float] = Config.IOU_THRESHOLD
):
    """
    Synchronous analysis endpoint that processes an uploaded video and returns
    bird counts, tracking samples, and weight estimates in a single response.
    """

    # Validate file type
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    file_ext = Path(video.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )

    # Check file size
    video.file.seek(0, 2)
    file_size = video.file.tell()
    video.file.seek(0)

    max_size_bytes = Config.MAX_VIDEO_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {Config.MAX_VIDEO_SIZE_MB}MB"
        )

    # Save to a temporary file under the upload directory
    Config.create_directories()
    temp_dir = Config.UPLOAD_DIR / "sync"
    temp_dir.mkdir(exist_ok=True)
    temp_video_path = temp_dir / video.filename

    try:
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Process video synchronously
        processor = VideoProcessor(
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            fps_sample=fps_sample,
        )
        # Use a dedicated output directory per run
        output_dir = Config.OUTPUT_DIR / "sync_outputs"
        output_dir.mkdir(exist_ok=True)

        results = processor.process_video(str(temp_video_path), str(output_dir))

        # Map internal results to the API schema
        weight_block = results.get("weight_estimates", {}) or {}
        artifacts = results.get("artifacts", {}) or {}

        return AnalysisResult(
            counts=results.get("counts", []),
            tracks_sample=results.get("tracks_sample", []),
            weight_estimates=weight_block.get("aggregate_weight", {}),
            confidence=weight_block.get("confidence_stats", {}),
            artifacts=artifacts,
            metadata=results.get("metadata", {}),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze video: {str(e)}"
        )
    finally:
        # Best-effort cleanup of the uploaded temp file
        if temp_video_path.exists():
            try:
                temp_video_path.unlink()
            except Exception:
                pass

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a processing task."""
    
    if task_id not in processing_tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )
    
    task = processing_tasks[task_id]
    
    # Return task status without sensitive file paths
    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
        "created_at": task["created_at"],
        "parameters": task["parameters"]
    }
    
    # Add results if available
    if "results" in task:
        response["results_available"] = True
        response["artifacts"] = task["results"].get("artifacts", {})
    else:
        response["results_available"] = False
    
    return response

@app.get("/task_results/{task_id}")
async def get_task_results(task_id: str):
    """Get the analysis results for a completed task."""
    
    if task_id not in processing_tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )
    
    task = processing_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed. Current status: {task['status']}"
        )
    
    if "results" not in task:
        raise HTTPException(
            status_code=500,
            detail="Results not available"
        )
    
    results = task["results"]
    
    # Format response according to API requirements
    return AnalysisResult(
        counts=results.get("counts", []),
        tracks_sample=results.get("tracks_sample", []),
        weight_estimates=results.get("weight_estimates", {}),
        confidence=results.get("weight_estimates", {}).get("confidence_stats", {}),
        artifacts=results.get("artifacts", {}),
        metadata=results.get("metadata", {})
    )

@app.get("/download/{task_id}/{file_type}")
async def download_file(task_id: str, file_type: str):
    """Download processed files (annotated video or results JSON)."""
    
    if task_id not in processing_tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )
    
    task = processing_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed. Current status: {task['status']}"
        )
    
    if "results" not in task:
        raise HTTPException(
            status_code=500,
            detail="Results not available"
        )
    
    artifacts = task["results"].get("artifacts", {})
    
    if file_type == "video":
        file_path = artifacts.get("annotated_video")
        media_type = "video/mp4"
        filename = "annotated_video.mp4"
    elif file_type == "json":
        file_path = artifacts.get("results_json")
        media_type = "application/json"
        filename = "analysis_results.json"
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Use 'video' or 'json'"
        )
    
    if not file_path or not Path(file_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_type}"
        )
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )

@app.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its associated files."""
    
    if task_id not in processing_tasks:
        raise HTTPException(
            status_code=404,
            detail="Task not found"
        )
    
    task = processing_tasks[task_id]
    
    # Delete task directory and files
    task_dir = Path(task["output_dir"])
    if task_dir.exists():
        shutil.rmtree(task_dir)
    
    # Remove from memory
    del processing_tasks[task_id]
    
    return {"message": f"Task {task_id} deleted successfully"}

@app.get("/tasks")
async def list_tasks():
    """List all tasks and their status."""
    
    tasks = []
    for task_id, task in processing_tasks.items():
        tasks.append({
            "task_id": task_id,
            "status": task["status"],
            "progress": task["progress"],
            "message": task["message"],
            "created_at": task["created_at"]
        })
    
    return {"tasks": tasks}

async def process_video_task(
    task_id: str,
    video_path: str,
    output_dir: str,
    fps_sample: int,
    conf_thresh: float,
    iou_thresh: float
):
    """Background task to process video."""
    
    try:
        # Update task status
        processing_tasks[task_id]["status"] = "processing"
        processing_tasks[task_id]["message"] = "Initializing video processor..."
        processing_tasks[task_id]["progress"] = 10
        
        # Initialize video processor
        processor = VideoProcessor(
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            fps_sample=fps_sample
        )
        
        processing_tasks[task_id]["message"] = "Processing video..."
        processing_tasks[task_id]["progress"] = 30
        
        # Process video
        results = processor.process_video(video_path, output_dir)
        
        processing_tasks[task_id]["message"] = "Finalizing results..."
        processing_tasks[task_id]["progress"] = 90
        
        # Store results
        processing_tasks[task_id]["results"] = results
        processing_tasks[task_id]["status"] = "completed"
        processing_tasks[task_id]["message"] = "Processing completed successfully"
        processing_tasks[task_id]["progress"] = 100
        
    except Exception as e:
        # Handle errors
        processing_tasks[task_id]["status"] = "failed"
        processing_tasks[task_id]["message"] = f"Processing failed: {str(e)}"
        processing_tasks[task_id]["error"] = str(e)
        
        # Log error (in production, would use proper logging)
        print(f"Task {task_id} failed: {str(e)}")

if __name__ == "__main__":
    # Run directly with the already-imported FastAPI app.
    # Note: reload=True requires an import string, so we disable it here.
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=False,
        log_level="info"
    )
