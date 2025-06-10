import face_recognition
import os
import cv2
import numpy as np

KNOWN_DIR = os.path.join("pc", "known_faces")

def load_known_faces():
    known_encodings = []
    known_names = []

    for filename in os.listdir(KNOWN_DIR):
        if filename.endswith((".jpg", ".png")):
            path = os.path.join(KNOWN_DIR, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])

    return known_encodings, known_names

def detect_person(image_bytes, known_encodings, known_names):
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding)
        if True in matches:
            idx = matches.index(True)
            return known_names[idx]
        else:
            return "Unknown"
    return None
