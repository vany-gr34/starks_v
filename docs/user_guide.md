# EagleScanAI User Guide

## Overview

EagleScanAI is an industrial defect detection system that uses advanced computer vision (YOLOv11) to identify surface defects in manufacturing materials. The system provides both a web interface and programmatic access for defect detection.

## Getting Started

### System Requirements
- Python 3.8 or higher
- Webcam (for live detection)
- Modern web browser
- 4GB+ RAM recommended

### Installation

1. **Download/Clone the Project**
```powershell
git clone https://github.com/your-repo/eaglescan-ai
cd eaglescan-ai
```

2. **Set Up Environment**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. **Start the Application**
```powershell
python app.py
```

4. **Open Web Interface**
   - Navigate to: `http://localhost:8080`
   - The web interface will load automatically

## Supported Defect Types

The system can detect 6 types of surface defects:

| Defect Type | Description | Common in |
|------------|-------------|----------|
| **Crazing** | Fine cracks on the surface | Metal sheets, ceramics |
| **Inclusion** | Foreign material embedded | Steel, aluminum |
| **Patches** | Irregular surface patches | Coated materials |
| **Pitted Surface** | Small holes or depressions | Corroded metals |
| **Rolled-in Scale** | Scale marks from rolling | Hot-rolled steel |
| **Scratches** | Linear surface damage | Machined parts |

## Using the Web Interface

### Main Dashboard
The web interface (`http://localhost:8080`) provides an intuitive drag-and-drop interface for defect detection.

### Image Upload Detection

1. **Upload Image**:
   - Click "Choose File" or drag and drop an image
   - Supported formats: JPG, JPEG, PNG
   - Maximum file size: Recommended < 10MB

2. **Run Detection**:
   - Click "Analyze Image" button
   - Wait for processing (typically 2-5 seconds)

3. **View Results**:
   - Annotated image with bounding boxes
   - Defect list with confidence scores
   - Detection statistics

### Live Camera Detection

1. **Start Live Detection**:
   - Click "Start Live Camera" button
   - Allow camera access when prompted

2. **Real-time Analysis**:
   - Live video feed with defect annotations
   - Real-time detection statistics
   - FPS counter and detection count

3. **Stop Detection**:
   - Press 'Q' key or close window
   - Session summary will be displayed

### Web Interface Features

**Upload Zone**:
- Drag-and-drop functionality
- Visual feedback during upload
- Progress indicators

**Results Display**:
- Annotated images with colored bounding boxes
- Confidence scores for each detection
- Defect type labels
- Summary statistics

**Interactive Elements**:
- Responsive design for mobile/tablet
- Real-time progress updates
- Error handling with user feedback

## Command Line Usage

### Batch Image Processing

Use the standalone prediction script:

```powershell
cd src
python predict.py
```

**Custom Parameters**:
```python
from src.predict import DefectDetector

detector = DefectDetector('models/best.pt')
detections = detector.predict_image('path/to/image.jpg', conf_threshold=0.3)

print(f"Found {len(detections)} defects:")
for det in detections:
    print(f"- {det['class']}: {det['confidence']:.2f}")
```

### Live Detection

Run terminal-based live detection:

```powershell
cd src
python live.py
```

**Command Line Options**:
```powershell
python live.py --model ../models/best.pt --conf 0.5 --source 0
```

**Parameters**:
- `--model`: Path to model weights (default: `../models/best.pt`)
- `--conf`: Confidence threshold (default: 0.50)
- `--source`: Camera source (0=default, 1=external)

### Terminal Live Detection Features

- **Real-time Statistics**: FPS, detection count
- **Detection Log**: Frame-by-frame detection info
- **Session Summary**: Total frames, average FPS
- **Keyboard Controls**: Press 'Q' to quit

## API Usage

### Health Check
```python
import requests

response = requests.get("http://localhost:8080/health")
print(response.json())
# Output: {"status": "healthy", "model_loaded": true, "model_path": "models/best.pt"}
```

### Image Detection
```python
import requests
import base64

# Encode image
with open("image.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Send for detection
response = requests.post(
    "http://localhost:8080/predict",
    json={"image": img_data}
)

result = response.json()
print(f"Found {result['count']} defects")
for detection in result['detections']:
    print(f"- {detection['class']}: {detection['confidence']:.2f}")
```

### Model Training
```python
# Trigger training via API
response = requests.get("http://localhost:8080/train")
print(response.json())
```

## Understanding Results

### Detection Output

Each detection includes:

```json
{
    "class": "scratches",
    "confidence": 0.85,
    "bbox": [100, 150, 300, 250]
}
```

- **class**: Type of defect detected
- **confidence**: Detection confidence (0.0-1.0)
- **bbox**: Bounding box coordinates [x1, y1, x2, y2]

### Confidence Scores

| Confidence Range | Interpretation |
|-----------------|----------------|
| 0.8 - 1.0 | High confidence - Likely defect |
| 0.5 - 0.8 | Medium confidence - Probable defect |
| 0.25 - 0.5 | Low confidence - Possible defect |
| < 0.25 | Very low - Often filtered out |

### Bounding Boxes

Color-coded bounding boxes in the web interface:
- Different colors for different defect types
- Thickness indicates confidence level
- Labels show defect type and confidence

## Configuration

### Model Settings

**Default Configuration** (in code):
- Model path: `models/best.pt`
- Confidence threshold: 0.50 (web), 0.25 (live)
- IoU threshold: 0.7
- Input image size: Auto-detected

### Adjusting Sensitivity

**Higher Confidence** (fewer false positives):
```python
detections = detector.predict_image('image.jpg', conf_threshold=0.7)
```

**Lower Confidence** (catch more defects):
```python
detections = detector.predict_image('image.jpg', conf_threshold=0.3)
```

### Dataset Configuration

For custom training, edit `config/data.yaml`:
```yaml
path: /path/to/your/dataset
train: train/images
val: val/images
nc: 6  # Number of classes
names: ['defect1', 'defect2', ...]  # Class names
```

## Troubleshooting

### Common Issues

**1. "Model not found" Error**
- Ensure `models/best.pt` exists in the project directory
- Check file permissions
- Download pre-trained weights if missing

**2. Camera Not Working**
- Check camera permissions in browser/system
- Try different camera source (--source 1)
- Ensure camera not used by another application

**3. Web Interface Not Loading**
- Check Flask server is running: `python app.py`
- Verify port 8080 is not blocked
- Check browser console for JavaScript errors

**4. Poor Detection Results**
- Ensure good image quality and lighting
- Try adjusting confidence threshold
- Check if image contains supported defect types

**5. Memory Issues**
- Close other applications
- Use smaller image sizes
- Restart the application

### Getting Help

**Debug Mode**:
```powershell
# Enable debug logging
python app.py
# Check console output for detailed error messages
```

**Check Model Loading**:
```python
from ultralytics import YOLO
model = YOLO('models/best.pt')
print("Classes:", model.names)
print("Model loaded successfully!")
```

**API Debugging**:
- Use browser developer tools (F12)
- Check Network tab for API responses
- Verify JSON format for requests

## Performance Tips

### Optimization

1. **Image Quality**:
   - Use well-lit, high-contrast images
   - Avoid blurry or low-resolution images
   - Ensure defects are visible

2. **Processing Speed**:
   - Use GPU if available (automatic detection)
   - Resize large images before processing
   - Close unnecessary applications

3. **Accuracy**:
   - Use appropriate confidence thresholds
   - Ensure good lighting conditions
   - Test with various image angles

### Best Practices

- **Lighting**: Consistent, diffused lighting
- **Distance**: Maintain consistent camera-to-object distance
- **Angle**: Perpendicular view for best results
- **Background**: Use contrasting backgrounds when possible
- **Focus**: Ensure images are sharp and in focus

## Advanced Features

### Custom Model Training

1. **Prepare Dataset**:
   - Organize images in YOLO format
   - Create train/val splits
   - Update `config/data.yaml`

2. **Start Training**:
   ```python
   from eaglescan.pipeline.training_pipeline import TrainPipeline
   pipeline = TrainPipeline()
   pipeline.run_pipeline()
   ```

3. **Monitor Progress**:
   - Check console logs
   - Training artifacts saved to `artifacts/`

### Integration with External Systems

The API can be integrated with:
- Quality control systems
- Manufacturing execution systems (MES)
- Industrial cameras and PLCs
- Custom applications via REST API

### Batch Processing

For processing multiple images:

```python
from src.predict import DefectDetector

detector = DefectDetector()
results = detector.predict_folder('/path/to/images/', conf_threshold=0.5)
```

## Support and Updates

- Check the GitHub repository for updates
- Report issues via GitHub Issues
- Refer to API documentation for developers
- Check development guide for customization