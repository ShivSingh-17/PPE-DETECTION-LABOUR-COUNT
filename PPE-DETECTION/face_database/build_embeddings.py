


import os
import pickle
from deepface import DeepFace

DB_PATH = r"C:\Users\SHIV\Desktop\AI_FACE_DASHBOARD_FINAL\face_database"
MODEL = "Facenet"

embeddings = {}

for person in os.listdir(DB_PATH):
    person_dir = os.path.join(DB_PATH, person)
    if not os.path.isdir(person_dir):
        continue

    reps = []

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL,
                enforce_detection=False
            )[0]["embedding"]

            reps.append(rep)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if reps:
        embeddings[person] = reps
        print(f"[OK] {person} embeddings saved")

with open("face_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("✅ ALL EMBEDDINGS GENERATED")