from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
import aiofiles
import os
from pathlib import Path
import shutil
from typing import Optional
import time

app = FastAPI(title="Lost Object Tracker API")

# Configure CORS - Allow frontend at localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for uploads
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Load YOLO model once at startup
try:
    model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

# Global webcam object (reused for efficiency)
camera = None


@app.get("/")
async def root():
    return {
        "message": "Lost Object Tracker API is running!",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "track": "/track?target_object=phone",
            "detect": "/detect (POST)",
            "upload": "/upload (POST)",
            "test_camera": "/test_camera"
        }
    }


@app.get("/health")
async def health_check():
    """Check API health and model status"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "yolo_model": "yolov8n.pt"
    }


def get_camera():
    """Initialize and return camera object (singleton pattern)"""
    global camera
    if camera is None or not camera.isOpened():
        # Try different camera backends for Windows
        backends = [
            (cv2.CAP_MSMF, "Media Foundation (MSMF)"),  # Best for Windows 10/11
            (cv2.CAP_ANY, "Default"),
        ]
        
        for backend, backend_name in backends:
            for index in [0, 1]:
                print(f"🔍 Trying camera index {index} with {backend_name}...")
                try:
                    camera = cv2.VideoCapture(index, backend)
                    if camera.isOpened():
                        # Test if we can actually read a frame
                        ret, test_frame = camera.read()
                        if ret:
                            # Set camera properties for better performance
                            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            camera.set(cv2.CAP_PROP_FPS, 30)
                            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                            print(f"✅ Camera opened successfully: index {index}, {backend_name}")
                            return camera
                    camera.release()
                except Exception as e:
                    print(f"❌ Failed with {backend_name} index {index}: {e}")
                    if camera:
                        camera.release()
        
        print("❌ No working camera found")
        camera = None
    return camera


@app.get("/test_camera")
async def test_camera():
    """Test endpoint to verify camera and YOLO detection work"""
    if not model:
        raise HTTPException(status_code=503, detail="YOLO model not loaded")
    
    try:
        cam = get_camera()
        if not cam.isOpened():
            raise HTTPException(status_code=500, detail="Cannot access camera")
        
        # Capture a single frame
        success, frame = cam.read()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to capture frame")
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Extract detected objects
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detections.append({
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0])
                })
        
        return {
            "success": True,
            "frame_shape": frame.shape,
            "detections_count": len(detections),
            "detections": detections
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")


@app.get("/list_cameras")
async def list_cameras():
    """List all available camera devices"""
    available_cameras = []
    
    # Try Media Foundation backend (best for Windows)
    for index in range(3):
        try:
            cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append({
                        "index": index,
                        "backend": "Media Foundation (MSMF)",
                        "status": "✅ Working",
                        "resolution": f"{frame.shape[1]}x{frame.shape[0]}"
                    })
                else:
                    available_cameras.append({
                        "index": index,
                        "backend": "Media Foundation (MSMF)",
                        "status": "⚠️ Opened but cannot read frames"
                    })
                cap.release()
        except Exception as e:
            pass
    
    return {
        "available_cameras": available_cameras,
        "count": len(available_cameras),
        "tip": "Use index 0 for built-in webcam, index 1 for external USB camera"
    }


def generate_frames(target_object: Optional[str] = None):
    """
    Generator function that yields MJPEG frames with YOLO detection.
    
    - Captures frames from webcam
    - Runs YOLOv8 detection
    - Draws bounding boxes:
        * GREEN box + "FOUND!" text if target object is detected
        * RED boxes for other objects
    - Maintains >15 FPS
    """
    cam = get_camera()
    if not cam.isOpened():
        # Return error frame instead of raising exception
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "ERROR: Cannot access camera", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(error_frame, "Check if camera is in use by another app", (50, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    
    # Normalize target object name for matching (lowercase, strip spaces)
    target_normalized = target_object.lower().strip() if target_object else None

    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    try:
        while True:
            success, frame = cam.read()
            if not success:
                print("⚠️ Failed to read frame from camera")
                break
            
            # Run YOLO detection with lower confidence threshold
            # Optimize: use imgsz=320 for faster inference (vs default 640)
            results = model(frame, conf=0.3, verbose=False, imgsz=320)
            
            found_target = False
            
            # Process detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract box coordinates and class info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    # Check if this is the target object
                    is_target = (target_normalized and 
                                class_name.lower() == target_normalized)

                    if is_target:
                        found_target = True
                        # Draw GREEN box for target object
                        color = (0, 255, 0)  # Green
                        thickness = 3
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                        # Add "FOUND!" text above the box
                        label = f"FOUND! {class_name} {confidence:.2f}"
                        label_y = y1 - 30 if y1 - 30 > 30 else y1 + 30
                        
                        # Draw text background
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                        )
                        cv2.rectangle(
                            frame,
                            (x1, label_y - text_height - 10),
                            (x1 + text_width + 10, label_y + 5),
                            color,
                            -1
                        )
                        
                        # Draw text
                        cv2.putText(
                            frame, label, (x1 + 5, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2
                        )
                    else:
                        # Draw RED box for other objects
                        color = (0, 0, 255)  # Red
                        thickness = 2
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                        
                        # Add label
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(
                            frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                        )

            # Add overlay status text
            status_text = f"Looking for: {target_object or 'None'}"
            cv2.putText(
                frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            
            if found_target:
                cv2.putText(
                    frame, "STATUS: FOUND!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
            
            # Calculate and display FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            cv2.putText(
                frame, f"FPS: {current_fps}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
            
            # Encode frame as JPEG with optimized quality (lower = faster)
            ret, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, 75,  # Reduced from 85 for speed
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
    except GeneratorExit:
        print("🔴 Client disconnected from stream")
    except Exception as e:
        print(f"❌ Error in frame generation: {e}")
    finally:
        # Don't release camera here - keep it open for next connection
        pass


@app.get("/track")
async def track_object(target_object: Optional[str] = Query(None, description="Name of object to find (e.g., 'phone', 'keys', 'bottle')")):
    """
    Real-time MJPEG video stream with YOLOv8 object detection.

    Query Parameters:
        - target_object: Name of the object to highlight (optional)

    Returns:
        - MJPEG stream with bounding boxes
        - Green boxes + "FOUND!" for target object
        - Red boxes for other detected objects

    Usage:
        <img src="http://localhost:8000/track?target_object=phone" />
    """
    if not model:
        raise HTTPException(status_code=503, detail="YOLO model not loaded")

    return StreamingResponse(
        generate_frames(target_object),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect objects in an uploaded image using YOLO
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run YOLO detection
        results = model(img)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()
                }
                detections.append(detection)
        
        return JSONResponse(content={
            "success": True,
            "detections": detections,
            "count": len(detections)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload and save an image file
    """
    try:
        file_path = UPLOAD_DIR / file.filename
        
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "path": str(file_path)
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.delete("/uploads/{filename}")
async def delete_file(filename: str):
    """
    Delete an uploaded file
    """
    try:
        file_path = UPLOAD_DIR / filename
        if file_path.exists():
            os.remove(file_path)
            return {"success": True, "message": f"File {filename} deleted"}
        else:
            raise HTTPException(status_code=404, detail="File not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting Lost Object Tracker API...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    print("📹 Video stream: http://localhost:8000/track?target_object=phone")
    print("\n🎯 Available object classes (YOLOv8):")
    if model:
        print("   ", ", ".join(list(model.names.values())[:20]), "... and more")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
