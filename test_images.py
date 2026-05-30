from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("../YOLO Weights/vehicleplate.pt")

img = cv2.imread("car_park.jpg")

results = model(img)

for r in results:
    boxes = r.boxes

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        conf = float(box.conf[0])
        cls = int(box.cls[0])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        cv2.putText(
            img,
            f"Plate {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
