from sort import Sort
from ultralytics import YOLO
import easyocr
import cv2
import cvzone
import numpy as np
import os
import re
import time

# =========================================================
# 1) CONFIG / PATHS
# =========================================================
VIDEO_PATH = "../Videos/car-video-2.mp4"

VEHICLE_MODEL_PATH = "../YOLO Weights/vehicle_detection.pt"
PLATE_MODEL_PATH = "../YOLO Weights/vehicle_plate_detection.pt"

OUTPUT_DIR = "saved_plates"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OCR confidence threshold for saving
OCR_SAVE_THRESHOLD = 0.60

# Vehicle classes from COCO
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike"}

# Counting line
limits = [420, 297, 673, 297]
totalCount = []

# =========================================================
# 2) LOAD MODELS
# =========================================================
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

reader = easyocr.Reader(['en'], gpu=False)

# Cache these once, not inside the loop
mask = cv2.imread("mask.png")
imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)

# To avoid saving the same plate many times
saved_plate_texts = set()


# =========================================================
# 3) HELPERS
# =========================================================
def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))


def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = clamp(x1, 0, w - 1)
    y1 = clamp(y1, 0, h - 1)
    x2 = clamp(x2, 0, w - 1)
    y2 = clamp(y2, 0, h - 1)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def enhance_plate(plate_bgr):
    """
    Basic enhancement before OCR.
    """
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])

    sharp = cv2.filter2D(thresh, -1, kernel)

    plate_resized = cv2.resize(
        sharp,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    return plate_resized


def clean_plate_text(text):
    if text is None:
        return ""

    text = text.upper().strip()
    text = text.replace(" ", "")
    text = re.sub(r"[^A-Z0-9-]", "", text)
    return text


def read_plate_easyocr(plate_img):
    results = reader.readtext(plate_img)

    if len(results) == 0:
        return None, 0.0

    best_text = ""
    best_conf = 0.0

    for detection in results:
        bbox, text, conf = detection
        if conf > best_conf:
            best_text = text
            best_conf = conf

    best_text = clean_plate_text(best_text)

    if not best_text:
        return None, 0.0

    return best_text, best_conf


def save_plate_image(plate_img, plate_text, track_id=None):
    """
    Save plate crop using the plate text as the filename.
    If the same plate is found again, add a suffix to avoid overwrite.
    """
    plate_text = clean_plate_text(plate_text)

    if not plate_text:
        return None

    filename = f"{plate_text}.jpg"
    file_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(file_path):
        suffix = f"_{track_id}" if track_id is not None else f"_{int(time.time() * 1000)}"
        file_path = os.path.join(OUTPUT_DIR, f"{plate_text}{suffix}.jpg")

    cv2.imwrite(file_path, plate_img)
    return file_path


# =========================================================
# 4) VIDEO INPUT
# =========================================================
cap = cv2.VideoCapture(VIDEO_PATH)

# =========================================================
# 5) MAIN PIPELINE
#    Vehicle Detection -> Tracking -> Plate Detection -> Crop -> Enhance -> OCR -> Save
# =========================================================
while True:
    success, img = cap.read()
    if not success:
        break

    # Optional overlay image
    if mask is not None:
        imgRegion = cv2.bitwise_and(img, mask)
    else:
        imgRegion = img.copy()

    if imgGraphics is not None:
        img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    # -----------------------------------------------------
    # 5.1 VEHICLE DETECTION
    # -----------------------------------------------------
    vehicle_results = vehicle_model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in vehicle_results:
        for box in r.boxes:
            xc, yc, w, h = box.xywh[0]
            x1 = int(xc - w / 2)
            y1 = int(yc - h / 2)
            x2 = int(xc + w / 2)
            y2 = int(yc + h / 2)

            conf = np.ceil(float(box.conf[0]) * 100) / 100
            cls = int(box.cls[0])
            currentClass = vehicle_model.names[cls]

            if currentClass in VEHICLE_CLASSES and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # -----------------------------------------------------
    # 5.2 TRACK VEHICLES
    # -----------------------------------------------------
    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for resultT in resultsTracker:
        x1, y1, x2, y2, vId = resultT
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        vId = int(vId)

        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(
            img,
            f"Vehicle ID: {vId}",
            (max(0, x1), max(35, y1 - 10)),
            scale=1,
            thickness=1,
            offset=3
        )

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        if limits[0] <= cx <= limits[2] and limits[1] - 20 <= cy <= limits[1] + 20:
            if vId not in totalCount:
                totalCount.append(vId)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # -------------------------------------------------
        # 5.3 PLATE DETECTION INSIDE EACH VEHICLE ROI
        # -------------------------------------------------
        vehicle_crop = safe_crop(img, x1, y1, x2, y2)
        if vehicle_crop is None:
            continue

        plate_results = plate_model(vehicle_crop, stream=True)

        for pr in plate_results:
            for pbox in pr.boxes:
                px1, py1, px2, py2 = pbox.xyxy[0]
                px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)

                pconf = float(np.ceil(pbox.conf[0].item() * 100)) / 100
                if pconf < 0.50:
                    continue

                # Convert plate coords from vehicle crop back to full frame
                fx1 = x1 + px1
                fy1 = y1 + py1
                fx2 = x1 + px2
                fy2 = y1 + py2

                cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (0, 255, 0), 3)
                cvzone.putTextRect(
                    img,
                    f"license-plate {pconf:.2f}",
                    (max(0, fx1), max(35, fy1 - 10)),
                    scale=1,
                    thickness=1,
                    colorB=(0, 255, 0),
                    colorT=(255, 255, 255),
                    colorR=(0, 255, 0)
                )

                # -------------------------------------------------
                # 5.4 CROP PLATE
                # -------------------------------------------------
                plate_crop = safe_crop(img, fx1, fy1, fx2, fy2)
                if plate_crop is None:
                    continue

                # -------------------------------------------------
                # 5.5 ENHANCE PLATE
                # -------------------------------------------------
                enhanced_plate = enhance_plate(plate_crop)

                # Optional preview windows
                # cv2.imshow("Plate Crop", plate_crop)
                # cv2.imshow("Enhanced Plate", enhanced_plate)

                # -------------------------------------------------
                # 5.6 OCR WITH EASYOCR
                # -------------------------------------------------
                plate_text, plate_conf = read_plate_easyocr(enhanced_plate)

                if plate_text:
                    print(f"Plate: {plate_text} | OCR Conf: {plate_conf:.2f} | Vehicle ID: {vId}")

                    cvzone.putTextRect(
                        img,
                        f"{plate_text} {plate_conf:.2f}",
                        (max(0, fx1), max(70, fy1 - 35)),
                        scale=1,
                        thickness=1,
                        colorB=(255, 0, 0),
                        colorT=(255, 255, 255),
                        colorR=(255, 0, 0)
                    )

                    # -------------------------------------------------
                    # 5.7 SAVE IMAGE USING PLATE NUMBER AS FILE NAME
                    # -------------------------------------------------
                    if plate_conf >= OCR_SAVE_THRESHOLD and plate_text not in saved_plate_texts:
                        saved_path = save_plate_image(plate_crop, plate_text, track_id=vId)
                        if saved_path:
                            saved_plate_texts.add(plate_text)
                            print(f"Saved: {saved_path}")

    # -----------------------------------------------------
    # 5.8 SHOW COUNT
    # -----------------------------------------------------
    cv2.putText(
        img,
        str(len(totalCount)),
        (255, 100),
        cv2.FONT_HERSHEY_PLAIN,
        5,
        (50, 50, 255),
        8
    )

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================================================
# 6) CLEANUP
# =========================================================
cap.release()
cv2.destroyAllWindows()
