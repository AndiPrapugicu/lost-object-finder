from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, FileResponse
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

# Load YOLO model once at startup
try:
    model = YOLO("yolov8n.pt")  # Using YOLOv8 nano model
    print("✅ YOLO model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None

# Global webcam object (reused for efficiency)
camera = None

# Mount static files directory
static_path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Mount uploads directory for serving processed files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# YOLO object categories organized by type
OBJECT_CATEGORIES = {
    "People & Animals": {
        "icon": "👥",
        "objects": ["person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]
    },
    "Vehicles": {
        "icon": "🚗",
        "objects": ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
    },
    "Street Objects": {
        "icon": "🏙️",
        "objects": ["traffic light", "fire hydrant", "stop sign", "parking meter", "bench"]
    },
    "Accessories": {
        "icon": "👜",
        "objects": ["backpack", "umbrella", "handbag", "tie", "suitcase"]
    },
    "Sports": {
        "icon": "⚽",
        "objects": ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
                   "skateboard", "surfboard", "tennis racket"]
    },
    "Kitchen": {
        "icon": "🍽️",
        "objects": ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"]
    },
    "Food": {
        "icon": "🍕",
        "objects": ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", 
                   "donut", "cake"]
    },
    "Furniture": {
        "icon": "🛋️",
        "objects": ["chair", "couch", "potted plant", "bed", "dining table", "toilet"]
    },
    "Electronics": {
        "icon": "💻",
        "objects": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone"]
    },
    "Appliances": {
        "icon": "🏠",
        "objects": ["microwave", "oven", "toaster", "sink", "refrigerator"]
    },
    "Indoor": {
        "icon": "🖼️",
        "objects": ["book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    }
}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    index_file = static_path / "index.html"
    if index_file.exists():
        with open(index_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Lost Object Finder</h1><p>Static files not found.</p>")


@app.get("/api")
async def api_root():
    """API information endpoint"""
    return {
        "message": "Lost Object Tracker API is running!",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "categories": "/categories",
            "track": "/track?target_object=phone",
            "detect": "/detect (POST)",
            "upload": "/upload (POST)",
            "test_camera": "/test_camera"
        }
    }


@app.get("/categories")
async def get_categories():
    """
    Get all object categories with their objects.
    This endpoint provides structured data for the frontend to display categories.
    """
    return {
        "success": True,
        "categories": OBJECT_CATEGORIES,
        "total_categories": len(OBJECT_CATEGORIES),
        "total_objects": sum(len(cat["objects"]) for cat in OBJECT_CATEGORIES.values())
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
            (cv2.CAP_DSHOW, "DirectShow"),  # More stable for Windows
            (cv2.CAP_MSMF, "Media Foundation (MSMF)"),  # Best for Windows 10/11
            (cv2.CAP_ANY, "Default"),
        ]
        
        for backend, backend_name in backends:
            for index in [0, 1]:
                print(f"🔍 Trying camera index {index} with {backend_name}...")
                try:
                    camera = cv2.VideoCapture(index, backend)

                    # Set properties BEFORE testing frame read - BALANCED for speed and quality
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Good resolution for detection
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Good resolution for detection
                    camera.set(cv2.CAP_PROP_FPS, 30)  # Standard FPS
                    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPEG for speed

                    # Only set buffer size for non-MSMF backends (causes issues with MSMF)
                    if backend != cv2.CAP_MSMF:
                        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if camera.isOpened():
                        # Give camera time to initialize
                        time.sleep(0.5)

                        # Test if we can actually read multiple frames
                        for _ in range(5):
                            ret, test_frame = camera.read()
                            if ret and test_frame is not None and test_frame.size > 0:
                                width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                print(f"✅ Camera opened successfully: index {index}, {backend_name} @ {width}x{height}")
                                return camera
                            time.sleep(0.1)

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
    
    # Parse multiple target objects (comma-separated)
    target_objects = []
    if target_object:
        target_objects = [obj.lower().strip() for obj in target_object.split(',')]

    print(f"📹 Starting video stream - Looking for: {target_objects}")

    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    found_objects = set()  # Track which target objects were found
    consecutive_errors = 0  # Track consecutive read errors
    frame_skip = 2  # Process every 2nd frame for balanced speed/accuracy
    frame_count = 0
    last_detection_results = None  # Cache last detection results

    try:
        while True:
            success, frame = cam.read()

            # Validate frame read
            if not success or frame is None or frame.size == 0:
                consecutive_errors += 1
                print(f"⚠️ Failed to read frame from camera (attempt {consecutive_errors})")

                # If too many consecutive errors, break
                if consecutive_errors > 10:
                    print("❌ Too many consecutive frame read errors, stopping stream")
                    break

                time.sleep(0.1)
                continue

            # Reset error counter on successful read
            consecutive_errors = 0

            # Validate frame dimensions
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"⚠️ Invalid frame shape: {frame.shape}")
                continue

            # Create a copy to avoid memory issues
            frame = frame.copy()

            frame_count += 1

            # OPTIMIZATION: Skip frames - only run YOLO every Nth frame
            if frame_count % frame_skip == 0:
                # Run YOLO detection with BALANCED optimization
                # imgsize 320 for good accuracy, conf 0.4 for better detection
                results = model(frame, conf=0.4, verbose=False, imgsz=320, device='cpu', half=False, max_det=30)
                last_detection_results = results
                found_objects.clear()  # Reset for each frame
            else:
                # Reuse last detection results for skipped frames
                results = last_detection_results if last_detection_results else []

            # Process detections - ONLY draw target objects for maximum speed
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract box coordinates and class info
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]

                    # Check if this is one of the target objects
                    is_target = class_name.lower() in target_objects if target_objects else False

                    # ONLY draw target objects - skip others for speed
                    if is_target:
                        found_objects.add(class_name.lower())
                        # Draw GREEN box for target object (thicker line)
                        color = (0, 255, 0)  # Green
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Add object name and confidence
                        label = f"{class_name.upper()} {confidence:.2f}"
                        # Background for text
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(frame, (x1, y1 - text_height - 8), (x1 + text_width, y1), color, -1)
                        cv2.putText(
                            frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                        )

            # Add overlay status - smaller
            if target_objects and found_objects:
                found_text = f"FOUND: {len(found_objects)}/{len(target_objects)}"
                cv2.putText(
                    frame, found_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
                )
            
            # Calculate and display FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Display FPS with smaller font
            cv2.putText(
                frame, f"FPS: {current_fps}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )
            
            # Encode frame as JPEG with BALANCED quality for speed and clarity
            ret, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, 55,  # Balanced quality
                cv2.IMWRITE_JPEG_OPTIMIZE, 0   # Disable optimization for speed
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
        import traceback
        print(f"❌ Fatal error in frame generation: {e}")
        print(traceback.format_exc())
        # Release and reset camera on fatal error
        if cam:
            cam.release()
        global camera
        camera = None
    finally:
        # Don't release camera here on normal exit - keep it open for next connection
        print("📹 Video stream ended")


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


@app.post("/detect_file")
async def detect_file(
    file: UploadFile = File(...),
    target_object: Optional[str] = Query(None, description="Comma-separated list of objects to detect")
):
    """
    Detect objects in uploaded image or video file.

    For images: Returns annotated image with bounding boxes
    For videos: Returns first frame with detections (later can be extended to full video)
    """
    if not model:
        raise HTTPException(status_code=503, detail="YOLO model not loaded")

    try:
        # Read file content
        contents = await file.read()
        file_extension = file.filename.lower().split('.')[-1]

        # Parse target objects
        target_objects = []
        if target_object:
            target_objects = [obj.lower().strip() for obj in target_object.split(',')]

        # Check if it's an image or video
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'webp']
        video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'webm']

        if file_extension in image_extensions:
            # Process image
            nparr = np.frombuffer(contents, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

            # Run YOLO detection
            results = model(frame, conf=0.4, verbose=False)

            # Process detections
            detections = []
            found_objects = set()

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]

                    # Check if it's a target object
                    is_target = class_name.lower() in target_objects if target_objects else True

                    detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                        "is_target": is_target
                    })

                    if is_target:
                        found_objects.add(class_name.lower())
                        # Draw GREEN box for target object
                        color = (0, 255, 0)
                    else:
                        # Draw BLUE box for other objects
                        color = (255, 0, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Add label
                    label = f"{class_name.upper()} {confidence:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (x1, y1 - text_height - 8), (x1 + text_width, y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Add summary text
            if target_objects:
                summary = f"Found: {len(found_objects)}/{len(target_objects)} target objects"
                cv2.putText(frame, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Encode annotated image
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                raise HTTPException(status_code=500, detail="Failed to encode image")

            # Save annotated image
            output_filename = f"detected_{file.filename}"
            output_path = UPLOAD_DIR / output_filename
            with open(output_path, 'wb') as f:
                f.write(buffer.tobytes())

            return JSONResponse(content={
                "success": True,
                "file_type": "image",
                "detections": detections,
                "count": len(detections),
                "found_target_objects": list(found_objects),
                "output_file": f"/uploads/{output_filename}"
            })

        elif file_extension in video_extensions:
            # Save video temporarily
            temp_video_path = UPLOAD_DIR / file.filename
            with open(temp_video_path, 'wb') as f:
                f.write(contents)

            # Open video
            cap = cv2.VideoCapture(str(temp_video_path))
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Invalid video file")

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Process video and create output
            output_filename = f"detected_{file.filename}"
            output_path = UPLOAD_DIR / output_filename

            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            all_detections = []
            frame_num = 0
            found_objects_total = set()

            # Process every Nth frame for speed
            process_every_n_frames = 5

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1

                # Only run detection every N frames
                if frame_num % process_every_n_frames == 0:
                    results = model(frame, conf=0.4, verbose=False)

                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = result.names[class_id]

                            is_target = class_name.lower() in target_objects if target_objects else True

                            if is_target:
                                found_objects_total.add(class_name.lower())
                                color = (0, 255, 0)
                            else:
                                color = (255, 0, 0)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            label = f"{class_name.upper()} {confidence:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            all_detections.append({
                                "frame": frame_num,
                                "class": class_name,
                                "confidence": confidence,
                                "is_target": is_target
                            })

                # Add frame counter
                cv2.putText(frame, f"Frame: {frame_num}/{frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                out.write(frame)

            cap.release()
            out.release()

            return JSONResponse(content={
                "success": True,
                "file_type": "video",
                "frames_processed": frame_num,
                "total_detections": len(all_detections),
                "unique_objects": len(set(d["class"] for d in all_detections)),
                "found_target_objects": list(found_objects_total),
                "output_file": f"/uploads/{output_filename}",
                "video_info": {
                    "fps": fps,
                    "frame_count": frame_count,
                    "resolution": f"{width}x{height}"
                }
            })

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload image (jpg, png) or video (mp4, avi, mov)")

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


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


@app.get("/stop_camera")
async def stop_camera():
    """
    Release camera resources
    """
    global camera
    try:
        if camera is not None and camera.isOpened():
            camera.release()
            camera = None
            print("📷 Camera released successfully")
            return {"success": True, "message": "Camera stopped"}
        else:
            return {"success": True, "message": "Camera was not running"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print("🚀 Starting Lost Object Tracker API...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API docs available at: http://localhost:8000/docs")
    print("📹 Video stream: http://localhost:8000/track?target_object=phone")
    print("\n🎯 Available object classes (YOLOv8):")
    if model:
        print("   ", ", ".join(list(model.names.values())[:20]), "... and more")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)