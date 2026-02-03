# EagleScanAI Development Guide

## Project Structure

```
eaglescan_ai/
├── app.py                  # Flask application entry point
├── train.py               # Training script
├── setup.py               # Package setup
├── requirements.txt       # Dependencies
├── Dockerfile            # Container configuration
├── config/
│   └── data.yaml         # Dataset configuration
├── data/                 # Input data storage
├── demo_data/            # Sample test images
├── models/               # Trained model weights
│   ├── best.pt          # Best model weights
│   └── last.pt          # Last checkpoint
├── templates/
│   └── index.html       # Web interface
├── src/                  # Inference scripts
│   ├── predict.py       # Batch prediction
│   └── live.py          # Live detection
├── eaglescan/           # Main package
│   ├── components/      # Pipeline components
│   ├── entity/          # Data classes
│   ├── constant/        # Configuration constants
│   ├── utils/           # Utility functions
│   ├── pipeline/        # Training pipeline
│   ├── logger/          # Logging setup
│   └── exception/       # Custom exceptions
├── research/            # Jupyter notebooks
├── evaluation/          # Model evaluation
└── docs/               # Documentation
```

## Core Components

### 1. Training Pipeline (`eaglescan/pipeline/training_pipeline.py`)

The training pipeline consists of three main stages:

```python
class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.model_trainer_config = ModelTrainerConfig()
```

**Pipeline Stages:**
1. **Data Ingestion** - Downloads and extracts dataset from Google Drive
2. **Data Validation** - Validates dataset structure (train/val/data.yaml)
3. **Model Training** - Trains YOLOv11 model with custom configuration

### 2. Data Ingestion (`eaglescan/components/data_ingestion.py`)

```python
class DataIngestion:
    def download_data(self) -> str:
        # Downloads ZIP from Google Drive using gdown
        
    def extract_zip_file(self, zip_file_path: str) -> str:
        # Extracts dataset to feature store
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        # Main ingestion workflow
```

**Features:**
- Google Drive integration using `gdown`
- Automatic extraction to configured paths
- Artifact tracking for downstream components

### 3. Data Validation (`eaglescan/components/data_validation.py`)

Validates the presence of required files:
- `train/` directory
- `val/` directory  
- `data.yaml` configuration file

### 4. Model Trainer (`eaglescan/components/model_trainer.py`)

```python
class ModelTrainer:
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        # Configure YOLOv5 model (legacy code)
        # Train with custom parameters
        # Save best weights
```

**Training Configuration:**
- Image size: 416x416
- Batch size: 16 (configurable)
- Epochs: 1 (configurable)
- Pretrained weights: `yolov11.pt`

### 5. Flask Application (`app.py`)

**Key Features:**
- Model loading at startup
- Base64 image encoding/decoding
- Real-time inference
- Live camera detection
- Health monitoring
- CORS support

**ClientApp Class:**
```python
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.model_path = "models/best.pt"
        self.model = None
        self.load_model()
```

### 6. Utility Functions (`eaglescan/utils/main_utils.py`)

```python
# YAML file operations
def read_yaml_file(file_path: str) -> dict
def write_yaml_file(file_path: str, content: object, replace: bool = False)

# Image encoding/decoding for web API
def decodeImage(imgstring, fileName)
def encodeImageIntoBase64(croppedImagePath)
```

### 7. Configuration Management

**Application Constants** (`eaglescan/constant/application.py`):
```python
APP_HOST = "0.0.0.0"
APP_PORT = 8080
```

**Training Pipeline Constants** (`eaglescan/constant/training_pipeline/__init__.py`):
```python
ARTIFACTS_DIR: str = "artifacts"
DATA_DOWNLOAD_URL: str = "https://drive.google.com/file/d/1qa1mzbwQ0yJ9AlfWGMMjPSqJL0jcIJj6/view?usp=sharing"
MODEL_TRAINER_PRETRAINED_WEIGHT_NAME: str = "yolov11.pt"
MODEL_TRAINER_NO_EPOCHS: int = 1
MODEL_TRAINER_BATCH_SIZE: int = 16
```

## Development Workflow

### 1. Environment Setup

```powershell
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset Configuration

Update `config/data.yaml` for your dataset:
```yaml
path: ../neu_dataset_yolo_ready
train: train/images
val: val/images
nc: 6
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
```

### 3. Training

```python
# Option 1: Via API
import requests
response = requests.get("http://localhost:8080/train")

# Option 2: Direct pipeline
from eaglescan.pipeline.training_pipeline import TrainPipeline
pipeline = TrainPipeline()
pipeline.run_pipeline()
```

### 4. Inference

**Batch Prediction:**
```python
from src.predict import DefectDetector

detector = DefectDetector('models/best.pt')
detections = detector.predict_image('path/to/image.jpg')
```

**Live Detection:**
```python
from src.live import live_detection
live_detection(model_path='models/best.pt', confidence=0.50)
```

**Web API:**
```python
# Start Flask server
python app.py
# Access http://localhost:8080
```

### 5. Model Evaluation

The `evaluation/` folder structure supports:
- `metrics/` - Performance metrics
- `visualizations/` - Detection visualizations

## Dependencies

**Core Libraries:**
```
ultralytics>=8.3.0    # YOLOv11 support
torch>=2.0.0          # PyTorch
torchvision>=0.15.0   # Vision utilities
opencv-python>=4.6.0  # Computer vision
Pillow>=10.0.0        # Image processing
```

**Web Framework:**
```
flask                 # Web framework
flask-cors            # CORS support
```

**Data Processing:**
```
numpy>=1.23.0         # Numerical computing
scipy>=1.4.1          # Scientific computing
PyYAML>=5.3.1         # YAML processing
gdown                 # Google Drive downloads
```

## Configuration Files

### Data Configuration (`config/data.yaml`)
Defines dataset paths and class information.

### Entity Classes (`eaglescan/entity/`)
- `config_entity.py` - Configuration dataclasses
- `artifacts_entity.py` - Artifact dataclasses

### Exception Handling
Custom exception class in `eaglescan/exception/` for consistent error handling.

### Logging
Centralized logging configuration in `eaglescan/logger/`.

## Testing

### Sample Images
Use images in `demo_data/` for testing detection functionality.

### Web Interface Testing
1. Start Flask app: `python app.py`
2. Open browser: `http://localhost:8080`
3. Upload test image and verify detection results

### API Testing
```python
import requests
import base64

# Test health endpoint
response = requests.get("http://localhost:8080/health")
assert response.json()['status'] == 'healthy'
```

## Docker Support

Dockerfile provided for containerized deployment:

```dockerfile
# Build container
docker build -t eaglescan-ai .

# Run container
docker run -p 8080:8080 eaglescan-ai
```

## Performance Optimization

1. **Model Loading**: Models loaded once at startup
2. **Inference Speed**: Optimized confidence and IoU thresholds
3. **Memory Management**: Automatic cleanup of temporary files
4. **Concurrent Processing**: Flask handles multiple requests

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `models/best.pt` exists
2. **Import errors**: Check `eaglescan` package installation
3. **CUDA issues**: Verify GPU setup for training
4. **Web interface**: Check Flask server is running on correct port

### Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model loading
from ultralytics import YOLO
model = YOLO('models/best.pt')
print(model.names)
```

## Future Development

### Planned Enhancements
1. **Testing Framework**: Add unit and integration tests
2. **Monitoring**: Implement model performance tracking  
3. **Deployment**: Add Kubernetes configurations
4. **API Docs**: Generate OpenAPI/Swagger documentation
5. **CI/CD**: Complete GitHub Actions workflows

### Code Quality
- Follow PEP 8 style guidelines
- Use type hints where applicable
- Implement comprehensive error handling
- Add docstrings for all public functions