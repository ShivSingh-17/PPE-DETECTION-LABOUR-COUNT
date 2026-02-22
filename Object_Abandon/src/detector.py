


# src/detector.py

from ultralytics import YOLO
from config import ALLOWED_CLASSES, CONFIDENCE_THRESHOLD

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]

        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            if cls_name in ALLOWED_CLASSES and conf > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "class": cls_name,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf
                })

        return detections