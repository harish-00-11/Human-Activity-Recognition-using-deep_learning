import os
import numpy as np
import cv2

class Parameters:
    def __init__(self):
        self.CLASSES = open("model/action_recognition.txt").read().strip().split("\n")
        self.ACTION_RESNET = 'model/resnet-34.onnx'
        self.YOLO_CFG = "model/yolov3.cfg"
        self.YOLO_WEIGHTS = "model/yolov3.weights"
        self.YOLO_CLASSES = "model/coco.names"
        self.IMAGE_DIR = "C:\\Users\\Harish\\Desktop\\human activity recognition\\images\\"
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112
param = Parameters()
print("[INFO] loading YOLO model...")
net_yolo = cv2.dnn.readNet(param.YOLO_WEIGHTS, param.YOLO_CFG)
# Gpu
net_yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net_yolo.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net_yolo.getUnconnectedOutLayers()]
print("[INFO] loading action recognition model...")
net_action = cv2.dnn.readNet(param.ACTION_RESNET)
for image_name in os.listdir(param.IMAGE_DIR):
    image_path = os.path.join(param.IMAGE_DIR, image_name)
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"[INFO] loading image: {image_path}")
        image = cv2.imread(image_path)
        frame_resized = cv2.resize(image, (420, 360))
        blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (512, 512), (0, 0, 0), True, crop=False)
        net_yolo.setInput(blob)
        yolo_outputs = net_yolo.forward(output_layers)
        boxes, confidences, class_ids = [], [], []
        for output in yolo_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:
                    center_x = int(detection[0] * frame_resized.shape[1])
                    center_y = int(detection[1] * frame_resized.shape[0])
                    w = int(detection[2] * frame_resized.shape[1])
                    h = int(detection[3] * frame_resized.shape[0])
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = box
            label = str(param.CLASSES[class_ids[i]])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        captures = [frame_resized] * param.SAMPLE_DURATION
        action_blob = cv2.dnn.blobFromImages(captures, 1.0, (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
                                             (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
        action_blob = np.transpose(action_blob, (1, 0, 2, 3))
        action_blob = np.expand_dims(action_blob, axis=0)
        net_action.setInput(action_blob)
        action_outputs = net_action.forward()
        action_label = param.CLASSES[np.argmax(action_outputs)]
        cv2.putText(image, action_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Human Activity Recognition on Image", image)
        cv2.waitKey(0)
cv2.destroyAllWindows()
