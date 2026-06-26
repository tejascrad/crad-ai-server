import cv2
import numpy as np
import base64
import re
from datetime import datetime

LOITERING_TIME_THRESHOLD = 5   # seconds
CONFIDENCE_THRESHOLD = 0.4
MOVEMENT_THRESHOLD = 80


class LoiteringDetector:

    def __init__(self, model):
        # Shares yolov8n model
        self.model = model
        # Per-session tracking state
        # { session_id: { track_id: { entry_time, last_pos } } }
        self.tracking = {}

    def get_session_state(self, session_id):
        if session_id not in self.tracking:
            self.tracking[session_id] = {}
        return self.tracking[session_id]

    def clear_session(self, session_id):
        if session_id in self.tracking:
            del self.tracking[session_id]

    def run(self, frame, session_id, filename, all_rois=None, prev_gray=None):
        # Parse timestamp from filename
        current_time = self._parse_timestamp(filename)
        if current_time is None:
            import time
            current_time = time.time()

        state = self.get_session_state(session_id)

        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        loitering_detected = False
        details_parts = []
        display = frame.copy()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                current_pos = (center_x, center_y)

                if track_id not in state:
                    state[track_id] = {
                        "entry_time": current_time,
                        "last_pos": current_pos
                    }
                else:
                    old_x, old_y = state[track_id]["last_pos"]
                    movement = abs(center_x - old_x) + abs(center_y - old_y)
                    if movement > MOVEMENT_THRESHOLD:
                        state[track_id]["entry_time"] = current_time
                    state[track_id]["last_pos"] = current_pos

                duration = current_time - state[track_id]["entry_time"]

                if duration > LOITERING_TIME_THRESHOLD:
                    color = (0, 165, 255)
                    label = "LOITERING " + str(track_id) + " " + str(int(duration)) + "s"
                    loitering_detected = True
                    details_parts.append("Loitering Person " + str(track_id) + " (" + str(int(duration)) + "s)")
                else:
                    color = (0, 255, 0)
                    label = "Person " + str(track_id) + " " + str(int(duration)) + "s"

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if loitering_detected:
            cv2.rectangle(display, (0, 0), (display.shape[1], 50), (0, 100, 255), -1)
            cv2.putText(display, "LOITERING DETECTED",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 3)

        _, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        ann_b64 = base64.b64encode(buf.tobytes()).decode()

        return {
            "status": "success",
            "loitering_detected": loitering_detected,
            "human_detected": False,
            "fire_detected": False,
            "crowd_detected": False,
            "violation_detected": False,
            "motion_detected": False,
            "details": " | ".join(details_parts),
            "annotated_image": ann_b64
        }

    def _parse_timestamp(self, filename):
        match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{6})", filename)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)
            dt = datetime.strptime(date_str + " " + time_str, "%Y-%m-%d %H%M%S")
            return dt.timestamp()
        return None