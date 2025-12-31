# file name : correctPhotoOrientationYolo.py
import cv2
import numpy as np
import os
from ultralytics import YOLO

INPUT_DIR = r"./content/sample_data"
OUTPUT_DIR = r"./content/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONF = 0.4
IMG_SIZE = 640

pose_model = YOLO("yolov8l-pose.pt")

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

def upright_score(img):
    res = pose_model.predict(img, imgsz=IMG_SIZE, conf=CONF, verbose=False)
    if res[0].keypoints is None:
        return -1
    kps = res[0].keypoints.xy.cpu().numpy()
    if len(kps) == 0:
        return -1
    kp = kps[0]
    nose = kp[0]
    l_sh = kp[5]
    r_sh = kp[6]
    if np.any(np.isnan([nose, l_sh, r_sh])):
        return -1
    shoulder_y = (l_sh[1] + r_sh[1]) / 2
    return shoulder_y - nose[1]

def infer_rotation_from_pose(img):
    candidates = [
        (img, 0),
        (rotate_image_no_crop(img, 90), 90),
        (rotate_image_no_crop(img, -90), -90),
        (rotate_image_no_crop(img, 180), 180),
    ]
    best_img = img
    best_score = -1
    for im, _ in candidates:
        score = upright_score(im)
        if score > best_score:
            best_score = score
            best_img = im
    return best_img

for fname in sorted(os.listdir(INPUT_DIR)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    img = cv2.imread(os.path.join(INPUT_DIR, fname))
    if img is None:
        continue
    rotated = infer_rotation_from_pose(img)
    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), rotated)

print("Done")