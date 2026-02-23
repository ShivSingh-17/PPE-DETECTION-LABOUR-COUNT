


import streamlit as st
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from deepface import DeepFace

# ---------------- PAGE ----------------
st.set_page_config(layout="wide")
st.title("Labour Attendance System")

# ---------------- LOAD MODELS ----------------
person_model = YOLO("models/yolov8s.pt")
ppe_model = YOLO("models/ppe_best.pt")
face_model = YOLO("models/Core_Model_1.pt")

PPE_CLASSES = ppe_model.names

# ---------------- LOAD EMBEDDINGS ----------------
with open("face_database/face_embeddings.pkl", "rb") as f:
    FACE_DB = pickle.load(f)

# ---------------- COSINE ----------------
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- FACE RECOGNITION ----------------
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

# ---------------- VEST COLOR ----------------
def classify_vest_color(roi):

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0,120,70])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + \
               cv2.inRange(hsv, lower_red2, upper_red2)

    lower_green = np.array([35,50,50])
    upper_green = np.array([85,255,255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(red_mask)
    green_pixels = cv2.countNonZero(green_mask)

    if red_pixels > green_pixels and red_pixels > 500:
        return "RED"
    elif green_pixels > red_pixels and green_pixels > 500:
        return "GREEN"
    else:
        return "UNKNOWN"

# ---------------- HELPER ----------------
def is_inside(person_box, ppe_box):
    px1, py1, px2, py2 = person_box
    gx1, gy1, gx2, gy2 = ppe_box

    cx = (gx1 + gx2) / 2
    cy = (gy1 + gy2) / 2

    return (px1 <= cx <= px2) and (py1 <= cy <= py2)

# ---------------- DATA STORES ----------------
identity_cache = {}
red_labours = {}
green_labours = {}

# ---------------- UI PLACEHOLDERS ----------------
camera_placeholder = st.empty()

col1, col2 = st.columns(2)
red_list_placeholder = col1.empty()
green_list_placeholder = col2.empty()

red_count = st.sidebar.empty()
green_count = st.sidebar.empty()

# ---------------- STREAM ----------------
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        st.error("Camera not working")
        break

    # PERSON TRACK
    person_results = person_model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.4
    )

    if person_results[0].boxes.id is None:
        camera_placeholder.image(frame, channels="BGR")
        continue

    person_boxes = person_results[0].boxes.xyxy.cpu().numpy()
    person_ids = person_results[0].boxes.id.cpu().numpy().astype(int)

    # PPE DETECT
    ppe_results = ppe_model(frame, conf=0.4)[0]
    ppe_boxes = ppe_results.boxes.xyxy.cpu().numpy()
    ppe_classes = ppe_results.boxes.cls.cpu().numpy().astype(int)

    for box, pid in zip(person_boxes, person_ids):

        x1, y1, x2, y2 = map(int, box)

        # ---- Identity Lock ----
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

        name = identity_cache[pid]
        display_name = name if name != "Unknown" else f"Unknown ({pid})"

        # ---- Vest Logic ----
        for p_box, cls_id in zip(ppe_boxes, ppe_classes):

            label = PPE_CLASSES[cls_id]

            if label == "vest" and is_inside(box, p_box):

                vx1, vy1, vx2, vy2 = map(int, p_box)
                vest_roi = frame[vy1:vy2, vx1:vx2]
                colour = classify_vest_color(vest_roi)

                if colour == "RED":
                    red_labours[pid] = display_name

                elif colour == "GREEN":
                    green_labours[pid] = display_name

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,display_name,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

    # DRAW PPE
    for p_box, cls_id in zip(ppe_boxes, ppe_classes):
        x1, y1, x2, y2 = map(int, p_box)
        label = PPE_CLASSES[cls_id]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
        cv2.putText(frame,label,(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

    # DISPLAY CAMERA
    camera_placeholder.image(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        channels="RGB"
    )

    # SIDEBAR COUNTS
    red_count.markdown(f"### 🟥 Total Red Labour: {len(red_labours)}")
    green_count.markdown(f"### 🟩 Total Green Labour: {len(green_labours)}")

    # BOTTOM LISTS
    red_list_placeholder.markdown("## 🟥 Red Labour Attendance")
    red_list_placeholder.write(list(red_labours.values()))

    green_list_placeholder.markdown("## 🟩 Green Labour Attendance")
    green_list_placeholder.write(list(green_labours.values()))