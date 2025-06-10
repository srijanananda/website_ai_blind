import time, base64, io
from threading import Thread
from django.shortcuts import render
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import KnownFace
from django.shortcuts import render
# ML imports
import cv2, numpy as np, face_recognition, pickle
import speech_recognition as sr
camera = cv2.VideoCapture(0)
# Load YOLOv3
net = cv2.dnn.readNet('pc/ml/yolov3.weights', 'pc/ml/yolov3.cfg')
with open('pc/ml/coco.names') as f:
    CLASS_NAMES = [line.strip() for line in f]
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Globals
running = False
mode = None  # 'object' or 'person'
last_announce = {'name': None, 'time': 0}
transcript = ''

# Load known faces
def load_known():
    db = KnownFace.objects.all()
    encs, names = [], []
    for k in db:
        encs.append(pickle.loads(k.encoding))
        names.append(k.name)
    return encs, names
known_encs, known_names = [], []

def gen_frames():
    cap = cv2.VideoCapture(0)
    while running:
        ret, frame = cap.read()
        if not ret: break
        _, buf = cv2.imencode('.jpg', frame)
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
    cap.release()

def inference_loop():
    global mode, transcript, known_encs, known_names, last_announce
    rec = sr.Recognizer()
    mic = sr.Microphone()
    while running:
        with mic as src:
            rec.adjust_for_ambient_noise(src)
            audio = rec.listen(src, phrase_time_limit=4)
        try:
            text = rec.recognize_google(audio).lower()
        except:
            continue
        transcript = text

        now = time.time()
        # Commands
        if 'detect object' in text and mode != 'object':
            mode, last_announce = 'object', {'name':None,'time':0}

        if 'person detection' in text and mode != 'person':
            known_encs, known_names = load_known()
            mode, last_announce = 'person', {'name':None,'time':0}

        if 'stop detection' in text:
            mode = None

        # Perform detection
        _, frame = cv2.VideoCapture(0).read()
        obj = None
        if mode == 'object':
            blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True,crop=False)
            net.setInput(blob)
            outs = net.forward(out_layers)
            boxes, confidences, classIDs = [],[],[]
            h,w,_=frame.shape
            for out in outs:
                for det in out:
                    scores = det[5:]
                    cid = np.argmax(scores)
                    if scores[cid] > 0.5:
                        centerx, centery = int(det[0]*w), int(det[1]*h)
                        obj = CLASS_NAMES[cid]; break
                if obj: break
        elif mode == 'person':
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, boxes)
            for enc in encs:
                matches = face_recognition.compare_faces(known_encs, enc)
                name = "Unknown"
                if True in matches:
                    name = known_names[matches.index(True)]
                obj = name
                break

        # Speak if new + 10s passed
        if obj and obj != last_announce['name'] and time.time() - last_announce['time'] > 10:
            last_announce = {'name':obj,'time':time.time()}
            transcript = transcript + f' | DETECTED: {obj}'
    print("Inference ended.")

# Views
def pc_dashboard(request): return render(request, 'pc/index.html', {})

@csrf_exempt
def start_system(request):
    global running
    if not running:
        running=True
        Thread(target=inference_loop, daemon=True).start()
    return JsonResponse({'status':'started'})

@csrf_exempt
def stop_system(request):
    global running
    running=False
    return JsonResponse({'status':'stopped'})

@csrf_exempt
def upload_face(request):
    name = request.POST.get('name')
    imgb = request.FILES['image'].read()
    arr = face_recognition.load_image_file(io.BytesIO(imgb))
    enc = face_recognition.face_encodings(arr)
    if enc:
        kf = KnownFace(name=name, image=request.FILES['image'], encoding=pickle.dumps(enc[0]))
        kf.save()
        return JsonResponse({'status':'saved'})
    return JsonResponse({'status':'fail'})

def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame')
def get_transcript(request):
    global transcript, last_announce
    return JsonResponse({
        'transcript': transcript,
        'detection': last_announce['name']
    })


def index(request):
    return render(request, 'pc/index.html')

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')