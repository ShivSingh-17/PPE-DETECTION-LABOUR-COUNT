


import cv2
import streamlit as st

from src.detector import ObjectDetector
from src.tracker import CentroidTracker
from src.registry import ObjectRegistry
from src.removal_logic import RemovalLogic
from src.abandonment_stub import AbandonmentStub

st.title("Factory Object Removal — Detection + Cases")

detector = ObjectDetector("models/yolov8s.pt")
tracker = CentroidTracker()
registry = ObjectRegistry()
removal_logic = RemovalLogic(registry)
abandoned_stub = AbandonmentStub()

cap = cv2.VideoCapture(0)

frame_window = st.empty()
event_panel = st.empty()

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)

    person_boxes = []
    object_boxes = []

    # Separate persons and objects
    for det in detections:
        if det["class"] == "person":
            person_boxes.append(det["bbox"])
        else:
            object_boxes.append(det["bbox"])

    # Update tracker on objects only
    tracked_objects = tracker.update(object_boxes)

    visible_ids = []

    # Draw PERSON detections
    for (x1, y1, x2, y2) in person_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "PERSON", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # Draw OBJECT detections + tracking
    for object_id, centroid in tracked_objects.items():

        cx, cy = centroid
        obj_id = f"obj_{object_id}"
        visible_ids.append(obj_id)

        # Simulate abandoned state
        abandoned = abandoned_stub.check_abandoned(obj_id)

        registry.update(obj_id, {
            "missing": 0,
            "person_nearby": len(person_boxes) > 0,
            "status": "abandoned" if abandoned else "active"
        })

        # Draw tracker ID
        cv2.circle(frame, (cx, cy), 5, (0,255,0), -1)
        cv2.putText(frame, obj_id, (cx-20, cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # Process removal logic
    alerts = removal_logic.process(visible_ids, frame)

    # Show frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame_rgb)

    # Show alerts
    for alert in alerts:
        event_panel.write(alert)

cap.release()