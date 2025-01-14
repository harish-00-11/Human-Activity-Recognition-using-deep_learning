import cv2
import numpy as np
from playsound import playsound

class Parameters:
    def __init__(self):
        self.CLASSES = open("model/action_recognition_kinetics.txt").read().strip().split("\n")
        self.ACTION_RESNET = 'model/resnet-34_kinetics.onnx'
        self.YOLO_CFG = "model/yolov3.cfg"
        self.YOLO_WEIGHTS = "model/yolov3.weights"
        self.YOLO_CLASSES = "model/coco.names"
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112

param = Parameters()

audio_people_detected = r"C:\Users\Harish\Desktop\human activity recognition\model\Audio\people detected 1.wav"
audio_people_not_detected = r"C:\Users\Harish\Desktop\human activity recognition\model\Audio\people cannot be detected 1.wav"
audio_action_detected = r"C:\Users\Harish\Desktop\human activity recognition\model\Audio\Action Detected 1.wav"
audio_action_undetermined = r"C:\Users\Harish\Desktop\human activity recognition\model\Audio\Action cannot be determin 1.wav"

print("[INFO] loading YOLO model...")
net_yolo = cv2.dnn.readNet(param.YOLO_WEIGHTS, param.YOLO_CFG)
net_yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net_yolo.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net_yolo.getUnconnectedOutLayers()]

print("[INFO] loading action recognition model...")
net_action = cv2.dnn.readNet(param.ACTION_RESNET)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    frame_resized = cv2.resize(frame, (550, 400))
    blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (512, 512), (0, 0, 0), True, crop=False)
    net_yolo.setInput(blob)
    yolo_outputs = net_yolo.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    person_detected = False
    for output in yolo_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Person class
                person_detected = True
                center_x = int(detection[0] * frame_resized.shape[1])
                center_y = int(detection[1] * frame_resized.shape[0])
                w = int(detection[2] * frame_resized.shape[1])
                h = int(detection[3] * frame_resized.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    if person_detected:
        playsound(audio_people_detected)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            label = str(param.CLASSES[class_ids[i]]) if class_ids[i] < len(param.CLASSES) else 'Person'
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        captures = [frame_resized] * param.SAMPLE_DURATION
        action_blob = cv2.dnn.blobFromImages(
            captures, 1.0, (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
            (114.7748, 107.7354, 99.4750), swapRB=True, crop=True
        )
        action_blob = np.transpose(action_blob, (1, 0, 2, 3))
        action_blob = np.expand_dims(action_blob, axis=0)
        net_action.setInput(action_blob)
        action_outputs = net_action.forward()[0]
        exp_scores = np.exp(action_outputs - np.max(action_outputs))
        probabilities = exp_scores / np.sum(exp_scores)
        max_prob = np.max(probabilities) * 100
        action_label = param.CLASSES[np.argmax(probabilities)] if max_prob > 30 else None
        if action_label:
            playsound(audio_action_detected)
            confidence_text = f"{action_label} ({max_prob:.2f}%)"
        else:
            playsound(audio_action_undetermined)
            confidence_text = "Action cannot be determined"
    else:
        playsound(audio_action_undetermined)  # No person detected, action cannot be determined
        confidence_text = "No person detected"
    cv2.putText(frame, confidence_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Human Activity Recognition Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
