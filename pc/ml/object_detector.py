import cv2
import numpy as np
import os

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

coco_path = os.path.join("pc", "ml", "coco.names")
cfg_path = os.path.join("pc", "ml", "yolov3.cfg")
weights_path = os.path.join("pc", "ml", "yolov3.weights")

# Load model
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]


with open(coco_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

last_announced = ""
last_time = 0

def detect_object(image_bytes):
    global last_announced, last_time

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w, h = int(detection[2]*width), int(detection[3]*height)
                x, y = int(center_x - w/2), int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    if len(indices) > 0:
        best_idx = indices[0][0]
        detected = classes[class_ids[best_idx]]
        return detected
    return None
