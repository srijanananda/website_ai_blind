import cv2
import face_recognition
from django.conf import settings
from pc.models import KnownFace
import pickle

def load_known_faces():
    known_encodings = []
    known_names = []

    for face in KnownFace.objects.all():
        try:
            encoding = np.frombuffer(face.encoding, dtype=np.float64)
            known_encodings.append(encoding)
            known_names.append(face.name)
        except:
            continue

    return known_encodings, known_names

def detect_person(frame, known_encs, known_names):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for (top, right, bottom, left), enc in zip(boxes, encodings):
        name = "Unknown"
        matches = face_recognition.compare_faces(known_encs, enc)
        if True in matches:
            name = known_names[matches.index(True)]
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return name, frame
    return None, frame
