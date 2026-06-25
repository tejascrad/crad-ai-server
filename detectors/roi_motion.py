import cv2
import numpy as np
import base64
from ultralytics import YOLO

YOLO_CONFIDENCE = 0.35
COLORS = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255),(255,255,0)]


class ROIMotionDetector:

    def __init__(self, model_path):
        print("Loading YOLO model...")
        self.model = YOLO(model_path)
        print("YOLO model loaded.")

    def _person_inside_roi(self, frame, roi_points):
        h, w = frame.shape[:2]
        scale = 2.0 if max(h, w) < 1920 else 1.5
        upscaled = cv2.resize(frame, (int(w*scale), int(h*scale)))
        results = self.model(upscaled, verbose=False, imgsz=1280)[0]
        roi_poly = np.array(roi_points, np.int32)
        confirmed = []
        for box in results.boxes:
            if int(box.cls[0]) != 0 or float(box.conf[0]) < YOLO_CONFIDENCE:
                continue
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            x1=int(x1/scale); y1=int(y1/scale)
            x2=int(x2/scale); y2=int(y2/scale)
            foot_x = (x1+x2)//2; foot_y = y2
            inside = cv2.pointPolygonTest(roi_poly,(float(foot_x),float(foot_y)),False) >= 0
            if not inside:
                cx=(x1+x2)//2; cy=(y1+y2)//2
                inside = cv2.pointPolygonTest(roi_poly,(float(cx),float(cy)),False) >= 0
            if inside:
                confirmed.append((x1,y1,x2,y2,float(box.conf[0])))
        return len(confirmed) > 0, confirmed

    def run(self, frame, all_rois, prev_gray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            return {"status":"success","motion_detected":False,
                    "human_detected":False,"details":"","annotated_image":""}
        display = frame.copy()
        human_detected = False
        details_parts = []
        alert_count = 0
        for roi_idx, roi_points in enumerate(all_rois):
            color = COLORS[roi_idx % len(COLORS)]
            human_found, person_boxes = self._person_inside_roi(frame, roi_points)
            if human_found:
                human_detected = True
                alert_count += 1
                details_parts.append(f"Human detected in ROI {roi_idx+1}")
            pts_arr = np.array(roi_points, np.int32)
            if human_found:
                overlay = display.copy()
                cv2.fillPoly(overlay,[pts_arr],color)
                display = cv2.addWeighted(overlay,0.15,display,0.85,0)
                cv2.polylines(display,[pts_arr],True,color,1)
                x,y = roi_points[0]
                cv2.putText(display,f"ROI {roi_idx+1}",(x,y),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
                for (x1,y1,x2,y2,conf) in person_boxes:
                    cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(display,f"Person {conf:.2f}",(x1,y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
                cv2.putText(display,f"HUMAN DETECTED - ROI {roi_idx+1}",
                            (30,50+alert_count*35),
                            cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        _,buf = cv2.imencode(".jpg",display,[cv2.IMWRITE_JPEG_QUALITY,85])
        ann_b64 = base64.b64encode(buf.tobytes()).decode()
        return {"status":"success","motion_detected":False,
                "human_detected":human_detected,
                "details":" | ".join(details_parts),
                "annotated_image":ann_b64}