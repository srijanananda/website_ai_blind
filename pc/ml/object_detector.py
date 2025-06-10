import cv2
import numpy as np

CFG_PATH = 'pc/ml/yolov3.cfg'
WEIGHTS_PATH = 'pc/ml/yolov3.weights'
NAMES_PATH = 'pc/ml/coco.names'

# Load YOLOv3 network and class names
net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
with open(NAMES_PATH, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

LAYER_NAMES = net.getUnconnectedOutLayersNames()

def detect_objects(frame, conf_threshold=0.5, nms_threshold=0.4):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(LAYER_NAMES)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                box = detection[0:4] * np.array([w, h, w, h])
                (cx, cy, bw, bh) = box.astype("int")
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)

                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    label = ""
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y, bw, bh) = boxes[i]
            color = (0, 255, 0)
            label = CLASSES[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            break  # Announce only one object for now

    return frame, label
