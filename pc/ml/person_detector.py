import cv2
import face_recognition
from django.conf import settings
from pc.models import KnownFace
import pickle
import numpy as np

def load_known_faces():
    encs, names = [], []
    for kf in KnownFace.objects.all():
        encs.append(pickle.loads(kf.encoding))
        names.append(kf.name)
    return encs, names

def detect_faces(frame, known_encs, known_names):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    label = ""
    for (top, right, bottom, left), encoding in zip(boxes, encodings):
        matches = face_recognition.compare_faces(known_encs, encoding)
        name = "Unknown"
        if True in matches:
            name = known_names[matches.index(True)]
        label = name

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        break  # Announce only one face for now

    return frame, label
