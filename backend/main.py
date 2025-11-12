from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
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

# Create static directory for HTML interface
STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files
try:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
except Exception as e:
    print(f"⚠️ Could not mount static files: {e}")

# Load YOLO model once at startup
try:
    model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

# Global webcam object (reused for efficiency)
camera = None

# Object synonyms for smart matching
# Maps user-friendly names to YOLO class names
OBJECT_SYNONYMS = {
    'phone': ['cell phone', 'mobile phone', 'smartphone'],
    'mobile': ['cell phone'],
    'smartphone': ['cell phone'],
    'tv': ['tv', 'television'],
    'television': ['tv'],
    'laptop': ['laptop', 'computer'],
    'computer': ['laptop', 'tv'],
    'remote': ['remote', 'tv remote'],
    'keys': ['key'],
    'glasses': ['sunglasses'],
    'bottle': ['bottle', 'water bottle'],
    'cup': ['cup', 'coffee cup'],
    'mug': ['cup'],
}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the HTML interface"""
    try:
        html_file = STATIC_DIR / "index.html"
        if html_file.exists():
            with open(html_file, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
    except Exception as e:
        print(f"Error serving HTML: {e}")

    # Fallback JSON response if HTML not found
    return JSONResponse(content={
        "message": "Lost Object Tracker API is running!",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "track": "/track?target_object=phone",
            "detect": "/detect (POST)",
            "upload": "/upload (POST)",
            "test_camera": "/test_camera"
        }
    })


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
                            # OPTIMIZED: Balanced resolution for quality and speed
                            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            camera.set(cv2.CAP_PROP_FPS, 30)
                            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
                            # Use MJPEG codec for faster frame grabbing
                            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                            # Disable auto-focus/exposure for consistency
                            camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
                            print(f"✅ Camera opened: index {index}, {backend_name}, 640x480@30fps")
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


def generate_frames(target_object: Optional[str] = None, performance_mode: bool = False):
    """
    Generator function that yields MJPEG frames with YOLO detection.
    
    Args:
        target_object: Name of object to search for
        performance_mode: If True, skip every 2 frames for 3x FPS boost

    - Captures frames from webcam
    - Runs YOLOv8 detection
    - Draws bounding boxes:
        * GREEN box + "FOUND!" text if target object is detected
        * RED boxes for other objects
    - Optimized for 20-30 FPS (or 30+ with performance_mode)
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
    
    # Parse multiple target objects (comma-separated)
    target_objects = []
    if target_object:
        # Split by comma and normalize each object
        target_objects = [obj.lower().strip() for obj in target_object.split(',') if obj.strip()]

    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    frame_count = 0
    last_detections = []
    last_frame_with_boxes = None

    try:
        while True:
            success, frame = cam.read()
            if not success:
                print("⚠️ Failed to read frame from camera")
                break
            
            frame_count += 1

            # OPTIMIZATION 2: In performance mode, run detection every 2nd frame
            # This gives 2x FPS boost while still detecting objects
            should_detect = not performance_mode or (frame_count % 2 == 0)

            if should_detect:
                # OPTIMIZATION 3: Use optimized YOLO settings
                # imgsz=416 is good balance, conf=0.5 reduces false positives
                results = model(frame, conf=0.45, verbose=False, imgsz=416, half=False)
                last_detections = results
            else:
                # Reuse last detection results for smoother display
                results = last_detections

            found_target = False
            found_objects = []  # Track which objects were found

            # Process detections - show ALL selected target objects
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract box coordinates and class info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    # Smart matching: check if ANY target matches the detected object
                    is_target = False
                    matched_target = None

                    if target_objects:
                        class_lower = class_name.lower()

                        for target_normalized in target_objects:
                            # Method 1: Exact match
                            if class_lower == target_normalized:
                                is_target = True
                                matched_target = target_normalized
                                break
                            # Method 2: Check synonyms dictionary
                            elif target_normalized in OBJECT_SYNONYMS:
                                if class_lower in OBJECT_SYNONYMS[target_normalized]:
                                    is_target = True
                                    matched_target = target_normalized
                                    break
                            # Method 3: Partial match (target is part of class name)
                            elif target_normalized in class_lower:
                                is_target = True
                                matched_target = target_normalized
                                break
                            # Method 4: Reverse partial match (class name is part of target)
                            elif class_lower in target_normalized:
                                is_target = True
                                matched_target = target_normalized
                                break

                    # ONLY draw if it's one of the target objects
                    if is_target:
                        found_target = True
                        if class_name not in found_objects:
                            found_objects.append(class_name)
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
                    # REMOVED: else block that drew red boxes for other objects
                    # Now only the target object is shown with green box

            # Add overlay status text (smaller font for speed)
            if target_objects:
                count = len(target_objects)
                status_text = f"Looking for: {count} object{'s' if count > 1 else ''}"
            else:
                status_text = "Looking for: None"
            cv2.putText(
                frame, status_text, (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
            
            if found_target:
                found_count = len(found_objects)
                found_text = f"FOUND {found_count}: {', '.join(found_objects[:3])}"
                if found_count > 3:
                    found_text += f" +{found_count - 3}"
                cv2.putText(
                    frame, found_text, (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                )
            
            # Calculate and display FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            cv2.putText(
                frame, f"FPS: {current_fps}", (5, frame.shape[0] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )
            
            # OPTIMIZATION 4: Balanced JPEG encoding
            ret, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, 80,  # Good balance quality/speed
                cv2.IMWRITE_JPEG_OPTIMIZE, 0   # Disable optimization for speed
            ])
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
    except GeneratorExit:
        print("🔴 Client disconnected from stream - releasing camera")
        # Release camera when client stops streaming
        global camera
        if camera is not None:
            camera.release()
            camera = None
            print("✅ Camera released successfully")
    except Exception as e:
        print(f"❌ Error in frame generation: {e}")
    finally:
        pass


@app.get("/track")
async def track_object(
    target_object: Optional[str] = Query(None, description="Name of object to find (e.g., 'phone', 'keys', 'bottle')"),
    performance_mode: bool = Query(False, description="Enable performance mode for 3x FPS boost")
):
    """
    Real-time MJPEG video stream with YOLOv8 object detection.

    Query Parameters:
        - target_object: Name of the object to highlight (optional)
        - performance_mode: Skip frames for 2x FPS boost (default: false)

    Returns:
        - MJPEG stream with bounding boxes (640x480 resolution)
        - Green boxes + "FOUND!" for target object
        - Red boxes for other detected objects
        - 15-20 FPS normally, 25-30 FPS with performance_mode=true

    Usage:
        <img src="http://localhost:8000/track?target_object=phone" />
        <img src="http://localhost:8000/track?target_object=phone&performance_mode=true" />
    """
    if not model:
        raise HTTPException(status_code=503, detail="YOLO model not loaded")

    return StreamingResponse(
        generate_frames(target_object, performance_mode),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/stop_camera")
async def stop_camera():
    """
    Stop the camera and release resources.
    Called when user clicks Stop button.
    """
    global camera
    try:
        if camera is not None:
            camera.release()
            camera = None
            print("✅ Camera stopped and released by user")
            return {"success": True, "message": "Camera stopped successfully"}
        else:
            return {"success": True, "message": "Camera was already stopped"}
    except Exception as e:
        print(f"❌ Error stopping camera: {e}")
        return {"success": False, "message": f"Error: {str(e)}"}


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
    import socket

    # Get local IP address for LAN access
    def get_local_ip():
        try:
            # Create a socket to determine the local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "127.0.0.1"

    local_ip = get_local_ip()

    print("🚀 Starting Lost Object Tracker API...")
    print("\n" + "="*60)
    print("📡 SERVER ADDRESSES:")
    print("="*60)
    print(f"  📍 Local (this computer):  http://localhost:8000")
    print(f"  📍 Local (this computer):  http://127.0.0.1:8000")
    print(f"  🌐 LAN (other devices):    http://{local_ip}:8000")
    print("="*60)
    print(f"\n📱 Access from phone/tablet: Open browser and go to:")
    print(f"   → http://{local_ip}:8000")
    print(f"\n📚 API Documentation: http://{local_ip}:8000/docs")
    print(f"📹 Direct video stream: http://{local_ip}:8000/track?target_object=phone")
    print("\n🎯 Available object classes (YOLOv8):")
    if model:
        print("   ", ", ".join(list(model.names.values())[:20]), "... and more")
    print("\n⚠️  Make sure all devices are on the same WiFi network!")
    print("="*60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
