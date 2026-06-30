from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import uvicorn
import numpy as np
import cv2
import json
import threading
from detectors.roi_motion import ROIMotionDetector
from detectors.fire import FireDetector
from detectors.crowd import CrowdDetector
from detectors.ppe import PPEDetector
from detectors.loitering import LoiteringDetector

app = FastAPI()

roi_detector = None
fire_detector = None
crowd_detector = None
ppe_detector = None
loitering_detector = None
model_ready = False

def load_models():
    global roi_detector, fire_detector, crowd_detector, ppe_detector, loitering_detector, model_ready
    print("Loading AI models - please wait...")
    from ultralytics import YOLO
    yolo_model = YOLO("models/yolov8n.pt")
    roi_detector = ROIMotionDetector("models/yolov8n.pt")
    fire_detector = FireDetector("models/fire_best.pt")
    crowd_detector = CrowdDetector(yolo_model)
    ppe_detector = PPEDetector("models/ppe_best.pt")
    loitering_detector = LoiteringDetector(yolo_model)
    model_ready = True
    print("All models ready!")

threading.Thread(target=load_models, daemon=True).start()

sessions = {}

@app.get("/health")
def health():
    return {
        "status": "ok" if model_ready else "loading",
        "model_ready": model_ready,
        "active_sessions": len(sessions),
        "session_ids": list(sessions.keys())
    }

@app.post("/init")
async def init(
    session_id: str = Form(...),
    rois_json: str = Form(...),
    detection_types: str = Form("human")
):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Models still loading.")
    rois = [[(int(p[0]),int(p[1])) for p in roi] for roi in json.loads(rois_json)]
    types = [t.strip() for t in detection_types.split(",") if t.strip()]
    sessions[session_id] = {
        "rois": rois,
        "prev_gray": None,
        "detection_types": types
    }
    return {
        "status": "ready",
        "session_id": session_id,
        "roi_count": len(rois),
        "detection_types": types
    }

@app.post("/detect")
async def detect(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    filename: str = Form(""),
    pc_name: str = Form(""),
    folder_path: str = Form(""),
    base_file_name: str = Form("")
):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Models still loading.")
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found. Call /init first.")

    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot read image")

    state = sessions[session_id]
    types = state.get("detection_types", ["human"])

    human_detected = False
    fire_detected = False
    crowd_detected = False
    ppe_violation = False
    loitering_detected = False
    details = ""
    annotated_image = ""

    # ROI Human Detection
    if "human" in types:
        roi_result = roi_detector.run(frame, state["rois"], state["prev_gray"])
        human_detected = roi_result.get("human_detected", False)
        if human_detected:
            details = roi_result.get("details", "")
            annotated_image = roi_result.get("annotated_image", "")

    # Fire Detection
    if "fire" in types:
        fire_result = fire_detector.run(frame.copy())
        fire_detected = fire_result.get("fire_detected", False)
        if fire_detected:
            details = fire_result.get("details", "")
            annotated_image = fire_result.get("annotated_image", "")

    # Crowd Detection
    if "crowd" in types:
        crowd_result = crowd_detector.run(frame.copy())
        crowd_detected = crowd_result.get("crowd_detected", False)
        if crowd_detected:
            details = crowd_result.get("details", "")
            annotated_image = crowd_result.get("annotated_image", "")

    # PPE Detection
    if "ppe" in types:
        ppe_result = ppe_detector.run(frame.copy())
        ppe_violation = ppe_result.get("violation_detected", False)
        if ppe_violation:
            details = ppe_result.get("details", "")
            annotated_image = ppe_result.get("annotated_image", "")

    # Loitering Detection
    if "loitering" in types:
        loiter_result = loitering_detector.run(frame.copy(), session_id, filename)
        loitering_detected = loiter_result.get("loitering_detected", False)
        if loitering_detected:
            details = loiter_result.get("details", "")
            annotated_image = loiter_result.get("annotated_image", "")

    # Update prev_gray
    sessions[session_id]["prev_gray"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    del frame, contents

    return {
        "status": "success",
        "human_detected": human_detected,
        "fire_detected": fire_detected,
        "crowd_detected": crowd_detected,
        "loitering_detected": loitering_detected,
        "violation_detected": ppe_violation,
        "motion_detected": False,
        "details": details,
        "annotated_image": annotated_image
    }

@app.post("/session/end")
async def end(session_id: str = Form(...)):
    loitering_detector.clear_session(session_id)
    sessions.pop(session_id, None)
    return {"status":"ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8765, reload=False)