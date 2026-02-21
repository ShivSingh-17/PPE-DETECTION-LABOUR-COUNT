


import streamlit as st
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from deepface import DeepFace

# ---------------- PAGE ----------------
st.set_page_config(layout="wide")
st.title("PPE + Identity Detection — MVP")

# ---------------- LOAD MODELS ----------------
person_model = YOLO("models/yolov8s.pt")
ppe_model = YOLO("models/ppe_best.pt")
face_model = YOLO("models/Core_Model_1.pt")

PPE_CLASSES = ppe_model.names

# ---------------- LOAD EMBEDDINGS ----------------
with open("face_database/face_embeddings.pkl", "rb") as f:
    FACE_DB = pickle.load(f)

# ---------------- COSINE FUNCTION ----------------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- RECOGNITION ----------------
def recognize_face(face_img):

    try:
        rep = DeepFace.represent(
            face_img,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]

    except:
        return "Unknown"

    best_name = "Unknown"
    best_score = 0

    for name, emb_list in FACE_DB.items():
        for emb in emb_list:
            score = cosine(rep, emb)

            if score > 0.45 and score > best_score:
                best_name = name
                best_score = score

    return best_name

# ---------------- IDENTITY CACHE ----------------
identity_cache = {}

# ---------------- PPE HELPER ----------------
def is_inside(person_box, ppe_box):
    px1, py1, px2, py2 = person_box
    gx1, gy1, gx2, gy2 = ppe_box

    cx = (gx1 + gx2) / 2
    cy = (gy1 + gy2) / 2

    return (px1 <= cx <= px2) and (py1 <= cy <= py2)

# ---------------- STREAM ----------------
cap = cv2.VideoCapture(0)

frame_placeholder = st.empty()
alert_placeholder = st.sidebar.empty()

while True:

    ret, frame = cap.read()
    if not ret:
        st.error("Camera not working")
        break

    alerts = []

    # -------- PERSON DETECT + TRACK --------
    person_results = person_model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.4
    )

    if person_results[0].boxes.id is None:
        frame_placeholder.image(frame, channels="BGR")
        continue

    person_boxes = person_results[0].boxes.xyxy.cpu().numpy()
    person_ids = person_results[0].boxes.id.cpu().numpy().astype(int)

    # -------- PPE DETECT --------
    ppe_results = ppe_model(frame, conf=0.4)[0]
    ppe_boxes = ppe_results.boxes.xyxy.cpu().numpy()
    ppe_classes = ppe_results.boxes.cls.cpu().numpy().astype(int)

    # -------- PER PERSON --------
    for box, pid in zip(person_boxes, person_ids):

        x1, y1, x2, y2 = map(int, box)

        # ===== FACE RECOGNITION =====
        if pid not in identity_cache:

            person_crop = frame[y1:y2, x1:x2]

            face_results = face_model(person_crop, conf=0.5)[0]

            if len(face_results.boxes) > 0:

                fx1, fy1, fx2, fy2 = map(
                    int,
                    face_results.boxes.xyxy[0].cpu().numpy()
                )

                face_crop = person_crop[fy1:fy2, fx1:fx2]

                name = recognize_face(face_crop)

            else:
                name = "Unknown"

            identity_cache[pid] = name

        else:
            name = identity_cache[pid]

        # ===== PPE LOGIC =====
        helmet_present = False
        vest_present = False

        for p_box, cls_id in zip(ppe_boxes, ppe_classes):

            if is_inside(box, p_box):

                label = PPE_CLASSES[cls_id]

                if label == "helmet":
                    helmet_present = True
                if label == "vest":
                    vest_present = True

        if not helmet_present:
            alerts.append(f"{name} → Helmet Missing")

        if not vest_present:
            alerts.append(f"{name} → Vest Missing")

        # ===== DRAW PERSON =====
        color = (0, 255, 0)

        if not helmet_present or not vest_present:
            color = (0, 0, 255)

        display_name = f"{name}" if name != "Unknown" else f"Unknown ({pid})"

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(
            frame,
            display_name,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # -------- DRAW PPE BOXES --------
    for p_box, cls_id in zip(ppe_boxes, ppe_classes):

        x1, y1, x2, y2 = map(int, p_box)
        label = PPE_CLASSES[cls_id]

        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,0),
            2
        )

    # -------- STREAMLIT DISPLAY --------
    frame_placeholder.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB"
    )

    if alerts:
        alert_placeholder.markdown("## Alerts")
        for a in alerts:
            alert_placeholder.error(a)
    else:
        alert_placeholder.success("All Compliant")