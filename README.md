# TransitLK - Sri Lankan Automatic Number Plate Recognition (ANPR)

## Overview

TransitLK is a computer vision-based Automatic Number Plate Recognition (ANPR) system designed for Sri Lankan vehicles.

The system uses state-of-the-art object detection, tracking, and Optical Character Recognition (OCR) technologies to:

- Detect vehicles in video streams
- Track vehicles across frames
- Detect Sri Lankan license plates
- Extract and recognize plate numbers
- Save detected plate images automatically
- Count vehicles crossing a predefined line
- Build a foundation for parking, toll, security, and traffic management systems

---

## Features

### Vehicle Detection
Uses a YOLO-based object detection model to detect:

- Car
- Bus
- Truck
- Motorbike

### Vehicle Tracking
Uses SORT (Simple Online Realtime Tracking) to:

- Assign unique IDs to vehicles
- Track vehicles across frames
- Prevent duplicate counting

### License Plate Detection
A custom-trained YOLO model detects Sri Lankan number plates.

### Plate Image Enhancement
Before OCR, the plate image is enhanced using:

- Grayscale conversion
- Bilateral filtering
- Adaptive thresholding
- Image sharpening
- Image upscaling

This improves OCR accuracy significantly.

### OCR Recognition
Uses EasyOCR to extract text from detected license plates.

### Automatic Plate Saving
Recognized plates are saved automatically:

```text
saved_plates/
├── CAB1234.jpg
├── WPABC5678.jpg
├── NCB9876.jpg
```

### Vehicle Counting
Counts vehicles crossing a virtual line.

### Real-Time Visualization
Displays:

- Vehicle bounding boxes
- Vehicle IDs
- Plate bounding boxes
- OCR results
- Vehicle count

---

## Project Architecture

```text
Video Input
      │
      ▼
Vehicle Detection (YOLO)
      │
      ▼
Vehicle Tracking (SORT)
      │
      ▼
Plate Detection (YOLO)
      │
      ▼
Crop Plate Region
      │
      ▼
Image Enhancement
      │
      ▼
EasyOCR
      │
      ▼
Plate Text Cleanup
      │
      ▼
Save Plate Image
      │
      ▼
Database / Future Applications
```

---

## Technologies Used

| Component | Technology |
|------------|------------|
| Object Detection | YOLO |
| Plate Detection | Custom YOLO Model |
| Tracking | SORT |
| OCR | EasyOCR |
| Image Processing | OpenCV |
| Visualization | CVZone |
| Programming Language | Python |

---

## Folder Structure

```text
TransitLK/
│
├── saved_plates/
│   ├── CAB1234.jpg
│   ├── WPABC5678.jpg
│
├── Images/
│   ├── car_park.jpg
│
├── Videos/
│   ├── car_video.mp4
│
├── YOLO Weights/
│   ├── vehicle_detection.pt
│   ├── vehicle_plate_detection.pt
│
├── sort.py
├── main.py
│
└── README.md
```

---

## Running the Project

Update the paths in `main.py` if necessary:

```python
VIDEO_PATH = "../Videos/car_video.mp4"

VEHICLE_MODEL_PATH = "../YOLO Weights/vehicle_detection.pt"

PLATE_MODEL_PATH = "../YOLO Weights/vehicle_plate_detection.pt"
```

Run:

```bash
python main.py
```

---

## Sample Output

### Vehicle Detection

```text
Vehicle ID: 12
```

### Plate Detection

```text
license-plate 0.92
```

### OCR Result

```text
CAB1234
```

### Console Output

```text
Plate: CAB1234 | OCR Conf: 0.93 | Vehicle ID: 12

Saved: saved_plates/CAB1234.jpg
```

---

## Future Improvements

### Database Integration

Store recognized vehicles:

```sql
CREATE TABLE vehicles (
    plate_number TEXT,
    timestamp DATETIME
);
```

### Entry / Exit Parking System

Track:

- Entry time
- Exit time
- Parking duration
- Parking fee

### Toll Collection System

Automatically:

- Detect vehicles
- Identify license plate
- Charge account

### Traffic Monitoring

Generate:

- Vehicle counts
- Peak traffic hours
- Vehicle statistics

### Super Resolution

Integrate:

- Real-ESRGAN
- ESRGAN

For improved plate recognition in:

- Low-resolution footage
- Highway cameras
- Night-time recordings

### Custom OCR Model

Train a dedicated OCR model for:

- Sri Lankan license plate fonts
- Government vehicle plates
- Special vehicle categories

### Cloud Deployment

Possible deployment using:

- FastAPI
- Docker
- PostgreSQL
- AWS
- Azure

---

## Applications

### Smart Parking Systems

Automatically record:

- Entry
- Exit
- Parking duration

### Toll Booth Automation

Detect and charge vehicles automatically.

### Security Monitoring

Identify vehicles entering restricted areas.

### Traffic Analytics

Monitor:

- Vehicle flow
- Traffic density
- Road usage

### Law Enforcement

Assist with:

- Stolen vehicle detection
- Vehicle tracking
- Traffic investigations

---

## Performance Notes

The system performance depends on:

- Camera quality
- Lighting conditions
- Video resolution
- OCR confidence threshold

For best results:

- Use HD or Full HD cameras
- Ensure plates are clearly visible
- Minimize motion blur
- Use daylight or well-lit environments

---
