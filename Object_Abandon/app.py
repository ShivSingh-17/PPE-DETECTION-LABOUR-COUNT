


# app.py

import cv2
import streamlit as st
from src.detector import ObjectDetector
from src.abandoned_logic import AbandonedLogic

st.title("Abandoned Object Detection — MVP")

detector = ObjectDetector("models/yolov8s.pt")
logic = AbandonedLogic()

cap = cv2.VideoCapture(0)

frame_placeholder = st.empty()

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    alerts = logic.update(None, detections)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["class"]

        color = (0, 255, 0)

        for alert_obj, _ in alerts:
            if det == alert_obj:
                color = (0, 0, 255)
                label = "ABANDONED"

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame)

cap.release()