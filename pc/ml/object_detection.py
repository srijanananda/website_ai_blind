import cv2
import numpy as np

# Paths to model files
cfg_path = 'pc/ml/yolov3.cfg'
weights_path = 'pc/ml/yolov3.weights'
names_path = 'pc/ml/coco.names'

# Load class names
with open(names_path, 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

# Load the network
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

# Use OpenCV's CPU backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def detect_objects_from_image_yolo(image_bytes, conf_threshold=0.5, nms_threshold=0.4):
    # Decode bytes to image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]

    # Create blob from image
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    ln = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []

    # Loop over each detection
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")

                # Get box coordinates
                x = int(centerX - w / 2)
                y = int(centerY - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-maxima suppression to suppress weaker overlapping boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(idxs) > 0:
        # Pick detection with highest confidence
        best_conf = 0
        best_class_id = None
        for i in idxs.flatten():
            if confidences[i] > best_conf:
                best_conf = confidences[i]
                best_class_id = class_ids[i]

        return CLASSES[best_class_id]
    else:
        return ""  # No object detected with high confidence
