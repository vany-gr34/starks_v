# EagleScanAI - Industrial Defect Detection System 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-2.0+-red.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A state-of-the-art computer vision system for real-time industrial defect detection using YOLO11 architecture. EagleScanAI provides both web-based and programmatic interfaces for defect detection in manufacturing environments.

## Project Overview

EagleScanAI is designed as a complete MLOps solution for industrial quality control. It combines computer vision, web technologies, and machine learning pipelines to deliver accurate, real-time defect detection capabilities.

### Supported Defect Types
The system detects 6 major types of surface defects commonly found in manufacturing:
- **Crazing**: Fine cracks on the surface
- **Inclusion**: Foreign material embedded in the surface
- **Patches**: Irregular surface patches
- **Pitted Surface**: Small holes or depressions
- **Rolled-in Scale**: Scale marks from rolling process
- **Scratches**: Linear surface damage

##  Architecture & Design Philosophy

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Layer   │    │  Service Layer  │    │   Core Layer    │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Web Interface  │    │ • Flask API     │    │ • YOLO11 Model  │
│ • REST API      │────▶│ • Route Handlers│────▶│ • Data Pipeline │
│ • Live Camera   │    │ • Image Encoding│    │ • Training Logic│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Design Principles

1. **Modular Architecture**: Each component has a single responsibility
2. **Scalable Pipeline**: Easy to extend with new defect types
3. **Real-time Performance**: Optimized for low-latency detection
4. **DevOps Ready**: Containerized and cloud-deployable
5. **Extensible Framework**: Plugin-based component system

### Core Components Logic

#### 1. **Training Pipeline** (`starks/pipeline/`)
**Purpose**: Orchestrates the complete model training workflow
**Logic**: 
- **Data Ingestion**: Downloads and validates training data from cloud sources
- **Data Validation**: Ensures data quality and format consistency
- **Model Training**: Fine-tunes YOLO11 on custom defect dataset
- **Artifact Management**: Tracks model versions and performance metrics

```python
# Training Flow
DataIngestion → DataValidation → ModelTrainer → ArtifactStorage
```

#### 2. **Component System** (`starks/components/`)
**Purpose**: Implements individual training stages as reusable components

**Data Ingestion Component**:
- Downloads datasets from Google Drive using authenticated APIs
- Implements retry logic and progress tracking
- Validates file integrity and format

**Data Validation Component**:
- Verifies YOLO annotation format
- Checks class distribution and data quality
- Generates validation reports

**Model Trainer Component**:
- Configures YOLO11 hyperparameters
- Implements custom loss functions for defect detection
- Handles GPU/CPU optimization

#### 3. **Web Application Layer** (`app.py`)
**Purpose**: Provides RESTful API and web interface
**Logic**:
- **Stateful Model Loading**: Singleton pattern ensures model loads once
- **Async Processing**: Non-blocking image processing
- **Error Handling**: Comprehensive exception management
- **Resource Management**: Automatic cleanup of temporary files

### API Architecture

```yaml
Endpoints:
  /predict:         # Image-based defect detection
    Input:          # Base64 encoded image
    Output:         # Annotated image + detection metadata
    Logic:          # YOLO inference → Post-processing → Response
  
  /live:            # Real-time camera detection
    Input:          # Camera stream (source=0)
    Output:         # Live detection window
    Logic:          # Stream processing → Real-time inference
  
  /train:           # Model training trigger
    Input:          # Training configuration
    Output:         # Training status
    Logic:          # Pipeline orchestration → Model artifacts
```

## Quick Start

### Prerequisites
```bash
# System Requirements
- Python 3.8+ (3.9+ recommended)
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ RAM
- Webcam (for live detection)
```

### Installation
```bash
# 1. Clone repository
git clone <repository-url>
cd starks_v

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "from ultralytics import YOLO; print('Setup complete!')"
```

### Quick Test
```bash
# Start web application
python app.py

# Test API health
curl http://localhost:8080/health

# Test model info
curl http://localhost:8080/model-info
```

##  Usage Guide

### 1. Web Interface Usage

**Starting the Application**:
```bash
python app.py
# Access: http://localhost:8080
```

**Web Features**:
- **Upload Interface**: Drag-and-drop image upload
- **Real-time Results**: Instant defect detection and visualization
- **Batch Processing**: Multiple image upload support
- **Detection Statistics**: Confidence scores and defect counts

### 2. API Integration

**Image Prediction**:
```python
import requests
import base64

# Encode image
with open('defect_image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

# Send prediction request
response = requests.post('http://localhost:8080/predict', 
                        json={'image': image_data})
result = response.json()

# Access results
print(f"Defects found: {result['count']}")
for detection in result['detections']:
    print(f"Type: {detection['class']}, Confidence: {detection['confidence']}")
```

**Live Detection**:
```python
import requests

# Start live camera detection
response = requests.get('http://localhost:8080/live')
# Camera window will open automatically
```

### 3. Standalone Scripts

**Batch Image Processing**:
```bash
# Process single image
python src/predict.py --image test_images/sample.jpg --conf 0.5

# Process entire folder
python src/predict.py --folder test_images/ --output results/
```

**Live Detection Script**:
```bash
# Start live detection with custom settings
python src/live.py --model weights/best.pt --conf 0.3 --camera 0
```

### 4. Custom Training

**Training New Models**:
```bash
# Using web API
curl -X GET http://localhost:8080/train

# Using Python script
python train.py --epochs 100 --batch-size 16 --img-size 640
```

**Training Configuration**:
```yaml
# data.yaml structure
path: ../dataset
train: train/images
val: val/images
nc: 6  # Number of classes
names: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
```

## 🔧 Configuration

### Model Configuration
```python
# starks/constant/application.py
APP_HOST = "0.0.0.0"    # Server host
APP_PORT = 8080         # Server port

# Detection settings
CONFIDENCE_THRESHOLD = 0.50  # Minimum confidence for detection
IOU_THRESHOLD = 0.7         # Intersection over Union threshold
```

### Environment Variables
```bash
# Optional environment configuration
export STARKS_MODEL_PATH="custom/path/to/model.pt"
export STARKS_DATA_URL="https://drive.google.com/your-dataset"
export STARKS_LOG_LEVEL="INFO"
```

## 📁 Project Structure Deep Dive

```
starks_v/
│
├── **Web Application Layer**
│   ├── app.py                    # Flask web server & API endpoints
│   ├── templates/
│   │   └── index.html           # Web interface
│   └── static/                  # CSS, JS, images
│
├──  **Core ML Package** (`starks/`)
│   ├── components/              # Training pipeline components
│   │   ├── data_ingestion.py   # Dataset download & extraction
│   │   ├── data_validation.py  # Data quality validation
│   │   └── model_trainer.py    # YOLO11 training logic
│   │
│   ├── pipeline/                # Orchestration logic
│   │   └── training_pipeline.py # End-to-end training workflow
│   │
│   ├── entity/                  # Data classes & configurations
│   │   ├── config_entity.py    # Configuration schemas
│   │   └── artifacts_entity.py # Training artifact definitions
│   │
│   ├── utils/                   # Utility functions
│   │   └── main_utils.py       # Image encoding/decoding utilities
│   │
│   ├── constant/                # Application constants
│   │   └── application.py      # Server & model configurations
│   │
│   ├── logger/                  # Logging system
│   └── exception/              # Custom exception handling
│
├──  **Standalone Scripts** (`src/`)
│   ├── predict.py              # Batch prediction utility
│   └── live.py                # Real-time detection utility
│
├──  **Model & Data**
│   ├── weights/                # Trained model weights
│   │   ├── best.pt            # Best model checkpoint
│   │   └── last.pt            # Latest training checkpoint
│   │
│   ├── test_images/           # Sample test images
│   └── data.yaml             # Dataset configuration
│
├──  **Deployment**
│   ├── Dockerfile            # Container definition
│   ├── requirements.txt      # Python dependencies
│   └── setup.py             # Package setup configuration
│
└──  **Documentation**
    ├── README.md            # This file
    ├── LICENSE             # MIT license
    └── reseach/            # Jupyter notebooks & experiments
```

### Component Responsibilities

| Component | Purpose | Key Files |
|-----------|---------|-----------|
| **Web Layer** | User interface & API | `app.py`, `templates/` |
| **Core ML** | Training & inference logic | `starks/` package |
| **Scripts** | Standalone utilities | `src/predict.py`, `src/live.py` |
| **Models** | AI artifacts | `weights/`, `data.yaml` |
| **Deployment** | Production setup | `Dockerfile`, `requirements.txt` |

##  Deployment Options

### Local Development
```bash
# Development server
python app.py
# Access: http://localhost:8080
```

### Docker Deployment
```bash
# Build container
docker build -t eaglescanai .

# Run container
docker run -p 8080:8080 -v $(pwd)/weights:/app/weights eaglescanai
```

### Cloud Deployment
```bash
# Azure Container Instances
az container create --resource-group myRG --name eaglescanai \
  --image your-registry/eaglescanai:latest --ports 8080

# AWS ECS / Google Cloud Run
# (Configure according to cloud provider documentation)
```

## 🔍 Troubleshooting

### Common Issues

**Model Loading Errors**:
```bash
# Verify model file exists
ls -la weights/best.pt

# Check model format
python -c "from ultralytics import YOLO; YOLO('weights/best.pt')"
```

**Camera Access Issues**:
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"
```

**Memory Issues**:
```python
# Reduce batch size in training
# Modify model_trainer.py:
model.train(data='data.yaml', batch=8)  # Reduce from 16 to 8
```

**Port Conflicts**:
```python
# Change port in starks/constant/application.py
APP_PORT = 8081  # Use different port
```

## 📈 Performance Optimization

### GPU Acceleration
```bash
# Verify CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Monitor GPU usage
nvidia-smi --loop=1
```

### Model Optimization
```python
# Use different YOLO variants for speed/accuracy tradeoff
YOLO('yolo11n.pt')  # Fastest (nano)
YOLO('yolo11s.pt')  # Balanced (small)
YOLO('yolo11m.pt')  # Accurate (medium)
YOLO('yolo11l.pt')  # Most accurate (large)
```

##Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black starks/ src/
flake8 starks/ src/
```

### Adding New Defect Types
1. Update `data.yaml` with new classes
2. Prepare training data in YOLO format
3. Retrain model with expanded dataset
4. Update web interface labels

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

##  Acknowledgments

- **Ultralytics YOLO** for the core detection framework
- **Flask** for the web application framework
- **OpenCV** for computer vision utilities
- **PyTorch** for deep learning infrastructure

## 📧 Support

For technical support or questions:
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@eaglescanai.com

---

**Made with ❤️ for industrial quality control**
