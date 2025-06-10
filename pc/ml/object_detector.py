import cv2
import numpy as np

cfg = 'pc/ml/yolov3.cfg'
weights = 'pc/ml/yolov3.weights'
names_path = 'pc/ml/coco.names'

with open(names_path) as f:
    CLASS_NAMES = [line.strip() for line in f]

net = cv2.dnn.readNetFromDarknet(cfg, weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

def detect_objects(frame):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                (cx, cy, w, h) = box.astype('int')
                x, y = int(cx - w/2), int(cy - h/2)
                label = CLASS_NAMES[class_id]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                return label, frame
    return None, frame
