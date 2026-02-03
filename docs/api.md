# EagleScanAI API Documentation

## Overview
The EagleScanAI Flask API provides endpoints for industrial defect detection using YOLOv11. The API runs on host `0.0.0.0` and port `8080` by default.

## Base URL
```
http://localhost:8080
```

## Supported Defect Types
The model detects 6 types of surface defects:
- `crazing` - Fine cracks on the surface
- `inclusion` - Foreign material embedded in the surface  
- `patches` - Irregular surface patches
- `pitted_surface` - Small holes or depressions
- `rolled-in_scale` - Scale marks from rolling process
- `scratches` - Linear surface damage

## Endpoints

### 1. Home Page
**GET** `/`

Returns the main web interface for image upload and detection.

**Response:**
- HTML page (`templates/index.html`)

---

### 2. Health Check
**GET** `/health`

Check API and model status.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true,
    "model_path": "models/best.pt"
}
```

---

### 3. Model Information
**GET** `/model-info`

Get information about the loaded model.

**Response:**
```json
{
    "model_type": "YOLO11n",
    "classes": ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"],
    "num_classes": 6,
    "status": "loaded"
}
```

---

### 4. Image Prediction
**POST** `/predict`

Detect defects in uploaded image.

**Headers:**
- `Content-Type: application/json`

**Request Body:**
```json
{
    "image": "<base64_encoded_image>"
}
```

**Response (Success):**
```json
{
    "image": "<base64_annotated_image>",
    "detections": [
        {
            "class": "scratches",
            "confidence": 0.85,
            "bbox": [100, 150, 300, 250]
        }
    ],
    "count": 1,
    "status": "success"
}
```

**Response (Error):**
```json
{
    "message": "Value not found inside json data",
    "status": "error"
}
```

**Detection Parameters:**
- Confidence threshold: `0.50`
- IoU threshold: `0.7`
- Input image saved as: `data/inputImage.jpg`
- Output saved to: `runs/detect/exp/`

---

### 5. Live Camera Detection
**GET** `/live`

Start live camera detection using webcam (source=0).

**Response:**
```json
{
    "message": "Camera started successfully!",
    "status": "success"
}
```

**Detection Parameters:**
- Confidence threshold: `0.25`
- IoU threshold: `0.7`
- Display: Real-time window with annotations
- Stream mode: Enabled

---

### 6. Training Pipeline
**GET** `/train`

Trigger the complete training pipeline.

**Response (Success):**
```json
{
    "message": "Training Successful!!",
    "status": "success"
}
```

**Response (Error):**
```json
{
    "message": "Training failed: <error_details>",
    "status": "error"
}
```

**Training Process:**
1. Data ingestion from Google Drive
2. Data validation (checks for train/val/data.yaml)
3. Model training with YOLOv11 architecture

## Error Codes

| Status Code | Description |
|-------------|-------------|
| 200 | Success |
| 400 | Bad Request (missing/invalid data) |
| 500 | Internal Server Error |

## Usage Examples

### Python Requests
```python
import requests
import base64

# Health check
response = requests.get("http://localhost:8080/health")
print(response.json())

# Image prediction
with open("image.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8080/predict",
    json={"image": img_data}
)
result = response.json()
print(f"Found {result['count']} defects")
```

### JavaScript/Fetch
```javascript
// Model info
fetch('/model-info')
    .then(response => response.json())
    .then(data => console.log(data));

// Image prediction
const imageFile = document.getElementById('fileInput').files[0];
const reader = new FileReader();
reader.onload = function(e) {
    const base64 = e.target.result.split(',')[1];
    
    fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({image: base64})
    })
    .then(response => response.json())
    .then(data => {
        console.log(`Detected ${data.count} defects`);
    });
};
reader.readAsDataURL(imageFile);
```

## Configuration

**Application Settings** (`eaglescan/constant/application.py`):
- Host: `0.0.0.0`
- Port: `8080`

**Model Settings**:
- Model path: `models/best.pt`
- Architecture: YOLOv11n
- Input image: `data/inputImage.jpg`

**Training Configuration** (`eaglescan/constant/training_pipeline/__init__.py`):
- Pretrained weights: `yolov11.pt`
- Epochs: `1` (default)
- Batch size: `16`
- Data source: Google Drive URL

## CORS Support
The API includes CORS support for cross-origin requests using Flask-CORS.

## Logging
All API operations are logged using Python's logging module at INFO level.