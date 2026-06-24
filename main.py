from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import uvicorn
import numpy as np
import cv2
import json
from detectors.roi_motion import ROIMotionDetector

app = FastAPI()

print("Loading AI model - please wait...")
detector = ROIMotionDetector("models/yolov8n.pt")
print("Server ready!")

sessions = {}

@app.get("/health")
def health():
    return {"status":"ok","active_sessions":len(sessions)}

@app.post("/init")
async def init(session_id: str = Form(...), rois_json: str = Form(...)):
    rois = [[(int(p[0]),int(p[1])) for p in roi] for roi in json.loads(rois_json)]
    sessions[session_id] = {"rois": rois, "prev_gray": None}
    return {"status":"ready","session_id":session_id,"roi_count":len(rois)}

@app.post("/detect")
async def detect(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    filename: str = Form(""),
    pc_name: str = Form(""),
    folder_path: str = Form(""),
    base_file_name: str = Form("")
):
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Session not found. Call /init first.")
    contents = await file.read()
    frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot read image")
    state = sessions[session_id]
    result = detector.run(frame, state["rois"], state["prev_gray"])
    sessions[session_id]["prev_gray"] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    del frame, contents
    return result

@app.post("/session/end")
async def end(session_id: str = Form(...)):
    sessions.pop(session_id, None)
    return {"status":"ok"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8765, reload=False)