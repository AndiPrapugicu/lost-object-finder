# 🔍 Lost Object Finder - YOLOv8 + FastAPI + React

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19+-blue.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9+-blue.svg)](https://www.typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Real-time object detection system that helps you find lost objects using your webcam and AI-powered computer vision.

![Demo Screenshot](https://via.placeholder.com/800x400.png?text=Lost+Object+Finder+Demo)

## ✨ Features

- 🎯 **Real-time Object Detection** - YOLOv8 powered detection at 15+ FPS
- 📹 **Live Camera Feed** - MJPEG streaming with minimal latency
- 🎨 **Modern UI** - Dark theme Material UI with smooth animations
- 🔍 **Smart Highlighting** - Green boxes for target objects, red for others
- ⚡ **Optimized Performance** - Efficient frame processing and encoding
- 🎮 **Easy Controls** - Start/stop camera, quick object suggestions
- 🌐 **Cross-platform** - Works on Windows, Mac, Linux

## 🎯 Project Overview

This is a Computer Vision project built with:
- **Backend**: FastAPI + OpenCV + Ultralytics YOLOv8
- **Frontend**: React + Vite + TypeScript + Material UI
- **Communication**: HTTP & MJPEG streaming

## 📸 Screenshots

| Feature | Screenshot |
|---------|------------|
| Main Interface | *Add your screenshot here* |
| Object Detection | *Add your screenshot here* |
| Camera Controls | *Add your screenshot here* |

## 🚀 Quick Start

### Backend Setup

1. Navigate to backend directory:
```powershell
cd backend
```

2. Install dependencies:
```powershell
pip install fastapi uvicorn opencv-python ultralytics python-multipart aiofiles torch torchvision
```

3. Start the server:
```powershell
python main.py
```

Backend will run at: **http://localhost:8000**
API docs at: **http://localhost:8000/docs**

### Frontend Setup

1. Navigate to frontend directory:
```powershell
cd frontend
```

2. Install dependencies:
```powershell
npm install
```

3. Start the dev server:
```powershell
npm run dev
```

Frontend will run at: **http://localhost:5173**

## 📡 API Endpoints

### `GET /`
Health check and available endpoints

### `GET /health`
Check API health and model status

### `GET /test_camera`
Test camera and YOLO detection (returns JSON with detections)

### `GET /track?target_object=<string>`
**Main endpoint** - Returns MJPEG video stream with object detection

**Query Parameters:**
- `target_object` (optional): Name of object to find (e.g., "phone", "keys", "bottle")

**Response:**
- MJPEG stream with bounding boxes
- **Green boxes** + "FOUND!" = Target object detected
- **Red boxes** = Other detected objects
- FPS counter and status overlay

**Usage Example:**
```html
<img src="http://localhost:8000/track?target_object=phone" />
```

## 🎨 Frontend Components

### VideoFeed.tsx

Main component for displaying video stream with overlays.

**Props:**
- `targetObject` (string): Name of object to search for
- `backendUrl` (optional): Backend API URL (default: http://localhost:8000)

**Features:**
- Real-time MJPEG streaming
- Animated overlays ("FOUND!", "Searching...")
- Error handling and loading states
- Dark mode Material UI styling
- Responsive design

## 🔧 Technical Implementation

### Backend (main.py)

**Key Functions:**

1. **`get_camera()`**
   - Singleton pattern for webcam access
   - Optimized settings (640x480, 30 FPS)

2. **`generate_frames(target_object)`**
   - Generator function for MJPEG streaming
   - Runs YOLOv8 detection on each frame
   - Draws bounding boxes:
     - Green (0, 255, 0) for target objects
     - Red (0, 0, 255) for other objects
   - Adds text overlays and FPS counter
   - Maintains >15 FPS performance

3. **CORS Configuration**
   - Allows connections from localhost:5173
   - Enables cross-origin streaming

### Frontend (VideoFeed.tsx)

**Features:**
- Material UI dark theme
- Pulse animation for "FOUND!" banner
- Loading spinner during connection
- Error alerts with troubleshooting tips
- Live status chip
- Responsive container

## 📦 Dependencies

### Backend
```txt
fastapi          # Web framework
uvicorn          # ASGI server
opencv-python    # Computer vision
ultralytics      # YOLOv8 implementation
python-multipart # File uploads
aiofiles         # Async file operations
torch            # PyTorch (required by ultralytics)
torchvision      # PyTorch vision utils
```

### Frontend
```json
{
  "@mui/material": "Material UI components",
  "@mui/icons-material": "Material UI icons",
  "react": "UI library",
  "axios": "HTTP client",
  "typescript": "Type safety",
  "vite": "Build tool"
}
```

## 🎯 Detectable Objects (YOLOv8)

YOLOv8n model can detect 80+ objects including:

**Common Items:**
- phone, laptop, mouse, keyboard
- bottle, cup, bowl, fork, knife, spoon
- book, scissors, remote, clock
- backpack, umbrella, handbag, tie, suitcase

**Electronics:**
- tv, laptop, mouse, keyboard, cell phone, remote

**Household:**
- chair, couch, bed, dining table, toilet
- oven, toaster, sink, refrigerator, microwave

[See full list in YOLO documentation]

## 🔍 Usage Flow

1. User enters object name (e.g., "phone")
2. Frontend sends request to `/track?target_object=phone`
3. Backend starts MJPEG stream with YOLOv8 detection
4. Each frame:
   - Captures from webcam
   - Runs YOLO detection
   - Filters for target object
   - Draws green box if found, red for others
   - Encodes as JPEG
   - Yields to stream
5. Frontend displays stream with overlays
6. "FOUND!" animation triggers when object detected

## 🎛️ Performance Optimization

- Camera resolution: 640x480 (balance of speed/quality)
- JPEG quality: 85% (reduces bandwidth)
- YOLOv8n model: Nano variant (fastest)
- Confidence threshold: 0.3 (catches more detections)
- Frame skipping: None (smooth stream)
- FPS target: >15 FPS (achieved in testing)

## 🐛 Troubleshooting

### Backend won't start
- Ensure Python 3.8+ is installed
- Check if camera is available: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
- Verify YOLO model downloaded: Check for `yolov8n.pt` in backend folder

### Frontend can't connect
- Verify backend is running at http://localhost:8000
- Check CORS configuration in main.py
- Test stream directly: Open http://localhost:8000/track?target_object=phone in browser

### Low FPS / Laggy stream
- Close other applications using camera
- Reduce camera resolution in `get_camera()`
- Use YOLOv8n (nano) instead of larger models
- Lower JPEG quality in `cv2.imencode()`

### Object not detected
- Ensure object is in YOLOv8 class list
- Try different lighting conditions
- Move object closer to camera
- Use exact YOLO class name (check model.names)

## 👥 Team Roles

**Andi (You):**
- ✅ YOLOv8 integration and detection logic
- ✅ MJPEG streaming implementation
- ✅ VideoFeed.tsx component
- ✅ CORS configuration
- ✅ /track endpoint
- ✅ Error handling

**Teammate 2:** [Database/Storage]
- User object preferences
- Detection history
- Image storage

**Teammate 3:** [Additional Features]
- Multi-object tracking
- Mobile responsiveness
- Sound alerts

## 📝 Code Structure

```
Lost Object Tracker/
├── backend/
│   ├── main.py              # FastAPI server + YOLO logic
│   ├── yolov8n.pt           # YOLOv8 nano model weights
│   ├── uploads/             # Uploaded images directory
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main application
│   │   ├── components/
│   │   │   └── VideoFeed.tsx # Video stream component
│   │   ├── App.css
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 🎓 Learning Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Material UI Components](https://mui.com/material-ui/getting-started/)
- [MJPEG Streaming Guide](https://blog.miguelgrinberg.com/post/video-streaming-with-flask)

## 📄 License

Educational project - 2025

---

Built with ❤️ by the Lost Object Finder Team
