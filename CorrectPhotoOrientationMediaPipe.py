# file name : CorrectPhotoOrientationMediaPipe.py
import cv2
import numpy as np
import mediapipe as mp
import math
import os

INPUT_DIR = r"./content/sample_data"
OUTPUT_DIR = r"./content/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def rotate_image_no_crop(image, angle):
    if angle == 0:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, M, (new_w, new_h))


for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, fname)
    img = cv2.imread(img_path)

    if img is None:
        continue

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    res = mp_face.process(rgb)

    if not res.multi_face_landmarks:
        continue

    lm = res.multi_face_landmarks[0].landmark

    le = np.array([lm[33].x * w, lm[33].y * h])
    re = np.array([lm[263].x * w, lm[263].y * h])
    nose = np.array([lm[1].x * w, lm[1].y * h])

    eye_center = (le + re) / 2

    vx = nose[0] - eye_center[0]
    vy = nose[1] - eye_center[1]

    angle_deg = math.degrees(math.atan2(vx, vy))

    if angle_deg > 45:
        rotate_deg = -90
    elif angle_deg < -45:
        rotate_deg = 90
    else:
        rotate_deg = 0

    rotated = rotate_image_no_crop(img, rotate_deg)

    out_path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(out_path, rotated)