import threading
import time
import cv2
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.conf import settings

from pc.ml.object_detector import detect_objects
from pc.ml.person_detector import detect_faces, load_known_faces
from pc.models import KnownFace

import pyttsx3

# Shared state
camera = cv2.VideoCapture(0)
frame_lock = threading.Lock()
output_frame = None
current_mode = None  # 'object' or 'person'
detection_result = ""
last_announced = ""
is_streaming = False

# TTS setup
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def tts_announce(text):
    global last_announced
    if text and text != last_announced:
        engine.say(text)
        engine.runAndWait()
        last_announced = text

def index(request):
    return render(request, 'pc/index.html')

@csrf_exempt
def start_stream(request):
    global is_streaming
    is_streaming = True
    return JsonResponse({"status": "streaming started"})

@csrf_exempt
def stop_stream(request):
    global is_streaming
    is_streaming = False
    return JsonResponse({"status": "streaming stopped"})

def generate_stream():
    global output_frame, is_streaming, current_mode, detection_result

    known_encs, known_names = [], []

    while True:
        if not is_streaming:
            time.sleep(0.1)
            continue

        ret, frame = camera.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        label = ""

        if current_mode == "object":
            frame, label = detect_objects(frame)
        elif current_mode == "person":
            if not known_encs:
                known_encs, known_names = load_known_faces()
            frame, label = detect_faces(frame, known_encs, known_names)

        if label:
            detection_result = label
            tts_announce(label)

        with frame_lock:
            output_frame = frame.copy()

        time.sleep(0.03)

def video_feed(request):
    def generate():
        global output_frame
        while True:
            with frame_lock:
                if output_frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', output_frame)
                if not ret:
                    continue
                frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
            time.sleep(0.05)

    return StreamingHttpResponse(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def get_transcript(request):
    return JsonResponse({
        'transcript': current_mode or '',
        'detection': detection_result
    })

@csrf_exempt
def start_system(request):
    t = threading.Thread(target=generate_stream, daemon=True)
    t.start()
    return JsonResponse({"status": "system thread started"})

@csrf_exempt
def set_mode(request):
    global current_mode, last_announced
    mode = request.GET.get("mode")
    if mode in ["object", "person"]:
        current_mode = mode
        last_announced = ""
        tts_announce(f"{mode} detection activated")
        return JsonResponse({"mode": mode})
    elif mode == "stop":
        current_mode = None
        last_announced = ""
        tts_announce("detection stopped")
        return JsonResponse({"mode": "stopped"})
    else:
        return JsonResponse({"error": "invalid mode"}, status=400)
