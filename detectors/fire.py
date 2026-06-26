import cv2
import numpy as np
import base64
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.3


class FireDetector:

    def __init__(self, model_path):
        print("Loading Fire model...")
        self.model = YOLO(model_path)
        print("Fire model loaded.")

    def run(self, frame, all_rois=None, prev_gray=None):
        fire_results = self.model(frame, verbose=False)
        fire_found = False
        details_parts = []
        display = frame.copy()

        for result in fire_results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                if class_name.lower() != "fire":
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                fire_found = True
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(display, "FIRE " + str(round(confidence, 2)),
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                details_parts.append("Fire detected conf=" + str(round(confidence, 2)))

        if fire_found:
            cv2.rectangle(display, (0, 0), (1280, 50), (0, 0, 200), -1)
            cv2.putText(display, "FIRE ALERT", (10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ann_b64 = base64.b64encode(buf.tobytes()).decode()

        return {
            "status": "success",
            "fire_detected": fire_found,
            "human_detected": False,
            "motion_detected": False,
            "details": " | ".join(details_parts),
            "annotated_image": ann_b64
        }