# Human-Activity-Recognition

This project implements a real-time human activity recognition system using deep learning models. The project integrates YOLO for object detection and ResNet-34 for action classification, offering a robust solution for applications like surveillance, healthcare monitoring, and more.

---

## Features
- Real-time person detection using **YOLO (You Only Look Once)**.
- Activity classification using a pre-trained **ResNet-34** model.
- Supports both **image-based** and **video-based** human activity recognition.
- GPU-accelerated performance with CUDA for faster inference.
- Audio feedback for detected actions and status.

---

## Folder Structure
```
model/
├── resnet-34.onnx                      
├── yolov3.cfg                          
├── yolov3.weights                      
├── coco.names                          
└── Audio/                              
images/
test/                                
train/
├── class 1                      
├── class 2                           
image.py                                
video_detection.py                      
realtime.py                             
```

---

## Installation

### Prerequisites
- Python 3.8 or later
- Libraries:
  - `opencv-python`
  - `numpy`
  - `playsound`
  - CUDA-compatible GPU and drivers for acceleration

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/harish-00-11/Human-Activity-Recognition-using-deep_learning.git
   cd real-time-human-activity-recognition
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the required pre-trained models:
   - Place the YOLO files (`yolov3.cfg`, `yolov3.weights`, and `coco.names`) in the `model/` directory.
   - Rename `resnet-34_kinetics.onnx` to `resnet-34.onnx` and place it in the same directory.
   
4. Add your test images to the `images/` folder and videos to the `test/` folder.

---

## Usage

### 1. Image-Based Recognition
Run the following command to classify actions from images:
```bash
python image.py
```

### 2. Video-Based Recognition
Process a video file to detect and classify actions:
```bash
python video_detection.py
```

### 3. Real-Time Recognition
Use a webcam to detect and classify actions in real-time:
```bash
python realtime.py
```

---

## Outputs
- **Bounding Boxes**: Persons detected by YOLO are highlighted with green rectangles.
- **Action Labels**: The identified activity is displayed on the screen with confidence percentages.
- **Audio Feedback**: Alerts indicating detection status and classified actions.

---

---

## Acknowledgments
- **YOLOv3**: For real-time object detection.
- **ResNet-34**: For robust activity recognition.
- **OpenCV**: For image processing and display.
- **CUDA**: For accelerating the computation.

---

