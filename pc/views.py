import cv2, time, threading
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import KnownFace
import pyttsx3

from pc.ml.object_detector import detect_objects
from pc.ml.person_detector import detect_person, load_known_faces

mode = None
running = False
streaming = False
camera = None
last_announce = {'name': None, 'time': 0}
known_encs, known_names = [], []
tts = pyttsx3.init()

def tts_announce(text):
    try:
        tts.say(text)
        tts.runAndWait()
    except Exception as e:
        print(f"TTS error: {e}")

def gen_frames():
    global camera, mode, last_announce, known_encs, known_names
    while streaming and camera and camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        label = None

        if mode == 'object':
            label, frame = detect_objects(frame)
        elif mode == 'person':
            label, frame = detect_person(frame, known_encs, known_names)

        if label and (label != last_announce['name'] or time.time() - last_announce['time'] > 10):
            last_announce = {'name': label, 'time': time.time()}
            tts_announce(label)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def pc_dashboard(request):
    return render(request, 'pc/index.html')

@csrf_exempt
def start_system(request):
    global running
    if not running:
        running = True
    return JsonResponse({'status': 'started'})

@csrf_exempt
def stop_system(request):
    global running
    running = False
    return JsonResponse({'status': 'stopped'})

@csrf_exempt
def start_stream(request):
    global streaming, camera
    if not streaming:
        camera = cv2.VideoCapture(0)
        streaming = True
    return JsonResponse({'status': 'streaming'})

@csrf_exempt
def stop_stream(request):
    global streaming, camera
    if streaming:
        streaming = False
        if camera: camera.release()
        camera = None
    return JsonResponse({'status': 'stopped'})

@csrf_exempt
def set_mode(request):
    global mode, known_encs, known_names
    m = request.GET.get('mode')
    if m == 'object':
        mode = 'object'
        tts_announce("Object detection started")
    elif m == 'person':
        known_encs, known_names = load_known_faces()
        mode = 'person'
        tts_announce("Person detection started")
    elif m == 'stop':
        mode = None
        tts_announce("Detection stopped")
    return JsonResponse({'status': 'mode_set', 'mode': mode})

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_transcript(request):
    return JsonResponse({'detection': last_announce['name']})
