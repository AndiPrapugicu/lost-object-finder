# 🚀 Quick Start Guide

## Start Backend (Terminal 1)

```powershell
cd backend
python main.py
```

✅ Backend running at: http://localhost:8000
📚 API docs at: http://localhost:8000/docs
🎥 Test stream: http://localhost:8000/track?target_object=phone

## Start Frontend (Terminal 2)

```powershell
cd frontend
npm run dev
```

✅ Frontend running at: http://localhost:5173

## Test the System

1. Open http://localhost:5173 in your browser
2. Type an object name (e.g., "phone", "bottle", "book")
3. Click "Search"
4. Point your camera at the object
5. Watch for the GREEN box when found! 🎯

## Common Objects to Try

- phone
- laptop
- mouse
- keyboard
- bottle
- cup
- book
- scissors
- remote
- backpack

## Troubleshooting

### Backend Issues
- Make sure camera is not in use by another app
- Check if yolov8n.pt exists in backend folder
- Verify Python 3.8+ is installed

### Frontend Issues
- Clear browser cache
- Check browser console for errors
- Make sure backend is running first

### Performance
- Close other camera apps
- Use good lighting
- Keep objects visible and centered

## Test Commands

### Test camera access:
```powershell
cd backend
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

### Test YOLO model:
```powershell
cd backend
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model OK'); print(list(model.names.values())[:10])"
```

### Test backend endpoint:
Open in browser: http://localhost:8000/health

### Test video stream:
Open in browser: http://localhost:8000/track?target_object=phone

---

Happy Object Hunting! 🔍✨
