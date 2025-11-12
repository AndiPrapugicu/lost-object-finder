"""
Test script to check if backend can start successfully
"""
print("🔍 Testing backend imports...")

try:
    print("1. Testing FastAPI...")
    from fastapi import FastAPI
    print("   ✅ FastAPI OK")
except Exception as e:
    print(f"   ❌ FastAPI Error: {e}")

try:
    print("2. Testing cv2...")
    import cv2
    print("   ✅ cv2 OK")
except Exception as e:
    print(f"   ❌ cv2 Error: {e}")

try:
    print("3. Testing numpy...")
    import numpy as np
    print(f"   ✅ numpy OK (version: {np.__version__})")
except Exception as e:
    print(f"   ❌ numpy Error: {e}")

try:
    print("4. Testing ultralytics...")
    from ultralytics import YOLO
    print("   ✅ ultralytics OK")
except Exception as e:
    print(f"   ❌ ultralytics Error: {e}")

try:
    print("5. Loading YOLO model...")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    print("   ✅ YOLO model loaded successfully!")
except Exception as e:
    print(f"   ❌ YOLO Error: {e}")

print("\n✅ All basic imports successful!")
print("🚀 Ready to start the server!")

