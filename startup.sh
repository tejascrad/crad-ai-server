#!/bin/bash
mkdir -p models
if [ ! -f models/yolov8n.pt ]; then
    echo "Downloading yolov8n.pt..."
    curl -L -o models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    echo "Model downloaded."
fi
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8765}