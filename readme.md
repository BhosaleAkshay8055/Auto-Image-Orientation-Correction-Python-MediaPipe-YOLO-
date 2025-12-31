# Auto Image Orientation Correction (Python | MediaPipe | YOLO)

> Automatically detect and fix incorrect photo orientation using **Python**, **MediaPipe**, **YOLOv8**, and **OpenCV**.

**GitHub Topics:**  
image-processing · computer-vision · image-rotation · opencv · mediapipe · yolo · python · ai

---

## 📷 Overview

This project provides **automatic image orientation correction** using two AI-based approaches:

- **MediaPipe Face Mesh** – fast and lightweight face-based orientation detection
- **YOLOv8 Pose Estimation** – robust full-body pose-based orientation detection

The scripts automatically:
- Read images from a folder
- Detect incorrect orientation
- Rotate images **without cropping**
- Save corrected images to an output directory

---

## 📁 Project Structure

.
├── CorrectPhotoOrientationMediaPipe.py
├── correctPhotoOrientationYolo.py
├── content/
│ ├── sample_data/ # Input images
│ └── output/ # Corrected images
└── README.md

markdown
Copy code

---

## 🔧 Script Details

### 1️⃣ CorrectPhotoOrientationMediaPipe.py

**Technology**
- MediaPipe Face Mesh
- OpenCV
- NumPy

**How It Works**
1. Loads images from `content/sample_data`
2. Detects face landmarks (eyes and nose)
3. Computes face direction using eye center → nose vector
4. Determines rotation (`-90°`, `+90°`, or `0°`)
5. Rotates image without cropping
6. Saves output to `content/output`

**Best For**
- Selfies
- Face-centric images
- Passport / ID photos
- Fast processing on low-resource machines

**Limitations**
- Face must be visible
- Not suitable for full-body images
- Does not handle 180° rotations

---

### 2️⃣ correctPhotoOrientationYolo.py

**Technology**
- YOLOv8 Pose Estimation
- OpenCV
- NumPy

**How It Works**
1. Loads YOLOv8 pose model
2. Detects nose and shoulder keypoints
3. Tests multiple rotations: `0°`, `90°`, `-90°`, `180°`
4. Calculates upright score (nose above shoulders)
5. Selects the best orientation
6. Saves corrected image to `content/output`

**Best For**
- Full-body photos
- Images without clear facial features
- High-accuracy orientation correction

**Limitations**
- Slower than MediaPipe
- YOLO model download required
- GPU recommended for large datasets

---

## 📦 Requirements

### Python Version
Python 3.9 – 3.11

markdown
Copy code

### Install Dependencies

**MediaPipe script**
```bash
pip install opencv-python mediapipe numpy
YOLO script

bash
Copy code
pip install opencv-python ultralytics numpy
YOLO will automatically download:

Copy code
yolov8l-pose.pt
▶️ Usage
Step 1: Add Images
Place your images in:

bash
Copy code
content/sample_data/
Supported formats:

Copy code
.jpg .jpeg .png
Step 2: Run Script
MediaPipe version

bash
Copy code
python CorrectPhotoOrientationMediaPipe.py
YOLO version

bash
Copy code
python correctPhotoOrientationYolo.py
Step 3: Output
Corrected images will be saved in:

bash
Copy code
content/output/
🆚 Comparison
Feature	MediaPipe	YOLO Pose
Detection Method	Face landmarks	Full-body pose
Speed	Fast	Slower
Face Required	Yes	No
180° Rotation	❌ No	✅ Yes
Accuracy	Medium	High
GPU	Not required	Optional

✅ Recommendation
Use MediaPipe for fast face-only orientation correction

Use YOLO for full-body or mixed datasets

YOLO provides the most reliable results across varied images

