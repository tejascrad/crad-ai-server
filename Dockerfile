FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models && \
    curl -L --fail -o models/yolov8n.pt \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt && \
    echo "yolov8n.pt size: $(wc -c < models/yolov8n.pt) bytes" && \
    curl -L --fail -o models/fire_best.pt \
    https://github.com/tejascrad/crad-ai-server/releases/download/v1.0/fire_best.pt && \
    echo "fire_best.pt size: $(wc -c < models/fire_best.pt) bytes"

EXPOSE 8765

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]