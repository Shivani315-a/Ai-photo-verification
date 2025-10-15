import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# ------------------ Model Setup ------------------
EYE_MODEL_PATH = "eye_status_cnn.h5"
EAR_MODEL_PATH = "ear_detection_model.h5"
GLASSES_MODEL_PATH = "eye-glasses_cnn.h5"

eye_model = load_model(EYE_MODEL_PATH)
ear_model = load_model(EAR_MODEL_PATH)
glasses_model = load_model(GLASSES_MODEL_PATH)

_, IMG_H, IMG_W, _ = eye_model.input_shape
IMG_SIZE = IMG_H

# Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# ------------------ Utilities ------------------
def prepare_image(img, size=IMG_SIZE):
    img_resized = cv2.resize(img, (size, size))
    img_array = img_resized.astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)


def safe_predict(model, img_array):
    pred = model.predict(img_array, verbose=0)
    return float(np.squeeze(pred))


# ------------------ Analysis Functions ------------------
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    return (x, y, w, h, roi_gray, roi_color)


def analyze_eyes(roi_gray, roi_color):
    eyes = eye_cascade.detectMultiScale(roi_gray)
    results = []

    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
            side = "left_eye" if i == 0 else "right_eye"
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            pred_val = safe_predict(eye_model, prepare_image(eye_img))
            status = "open" if pred_val > 0.65 else "closed"
            results.append({
                "side": side,
                "status": status,
                "confidence": round(pred_val, 4)
            })
    else:
        results.append({"error": "Could not detect both eyes"})

    overall = "unknown"
    if len(results) == 2 and all("status" in r for r in results):
        left_status = results[0]["status"]
        right_status = results[1]["status"]
        if left_status == "open" and right_status == "open":
            overall = "both_open"
        elif left_status == "closed" and right_status == "closed":
            overall = "both_closed"
        else:
            overall = f"{left_status}_{right_status}"

    return results, overall


def analyze_ears(img, face_box):
    x, y, w, h = face_box
    results = []
    offset_x = int(w * 0.10)
    ear_width = int(w * 0.20)
    ear_height = int(h * 0.5)
    ear_y = y + int(h * 0.25)

    ear_boxes = [
        (x - ear_width + offset_x, ear_y, ear_width, ear_height, "left_ear"),
        (x + w - offset_x, ear_y, ear_width, ear_height, "right_ear")
    ]

    for ex, ey, ew, eh, side in ear_boxes:
        ex, ey = max(0, ex), max(0, ey)
        ear_region = img[ey:ey+eh, ex:ex+ew]

        if ear_region.size == 0:
            results.append({"side": side, "visible": False, "confidence": None})
            continue

        ear_resized = cv2.resize(ear_region, (64, 64)) / 255.0
        ear_input = np.expand_dims(ear_resized, axis=0)
        pred_val = safe_predict(ear_model, ear_input)
        visible = pred_val >= 0.62
        results.append({
            "side": side,
            "visible": visible,
            "confidence": round(pred_val, 4)
        })

    return results


def analyze_glasses(roi_color):
    pred_val = safe_predict(glasses_model, prepare_image(roi_color))
    label = "no-glasses" if pred_val >= 0.6 else "glasses"
    return {"label": label, "confidence": round(pred_val, 4)}


def analyze_face(img):
    face_data = detect_face(img)
    if face_data is None:
        return {"error": "No face detected"}

    x, y, w, h, roi_gray, roi_color = face_data
    eyes, overall_eye_status = analyze_eyes(roi_gray, roi_color)
    ears = analyze_ears(img, (x, y, w, h))
    glasses = analyze_glasses(roi_color)

    return {
        "eyes": eyes,
        "overall_eye_status": overall_eye_status,
        "ears": ears,
        "glasses": glasses
    }
