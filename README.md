# EagleScanAI - Industrial Defect Detection System 🦅

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-2.0+-red.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A state-of-the-art computer vision system for real-time industrial defect detection using YOLOv11 architecture. EagleScanAI provides both web-based and programmatic interfaces for defect detection in manufacturing environments.

## 🎯 Project Overview

EagleScanAI is designed as a complete MLOps solution for industrial quality control. It combines computer vision, web technologies, and machine learning pipelines to deliver accurate, real-time defect detection capabilities for manufacturing environments.

### Supported Defect Types
The system detects 6 major types of surface defects commonly found in manufacturing:
- **Crazing**: Fine cracks on the surface
- **Inclusion**: Foreign material embedded in the surface
- **Patches**: Irregular surface patches
- **Pitted Surface**: Small holes or depressions
- **Rolled-in Scale**: Scale marks from rolling process
- **Scratches**: Linear surface damage

## 🏗️ Key Features

### Core Capabilities
- **Real-time Detection**: Live camera-based defect detection with instant feedback
- **Web Interface**: Intuitive drag-and-drop interface for image upload and analysis
- **REST API**: Complete API for integration with existing systems
- **Batch Processing**: Process multiple images efficiently
- **Custom Training**: Train on your own defect datasets
- **Health Monitoring**: Built-in system health checks and model status

### Technical Features
- **YOLOv11 Architecture**: Latest YOLO model for superior accuracy and speed
- **Multi-format Support**: JPG, PNG image formats supported
- **Confidence Scoring**: Adjustable confidence thresholds for detection sensitivity
- **Base64 Encoding**: Seamless image transfer via API
- **CORS Support**: Cross-origin requests enabled for web integration
- **Automated Pipeline**: Complete training pipeline with data validation
- **Google Drive Integration**: Automatic dataset download and management

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Modern web browser
- Webcam (for live detection)
- 4GB+ RAM recommended

### Installation

1. **Clone the Repository**
   - Download or clone the project to your local machine

2. **Set Up Environment**
   - Create a virtual environment
   - Install all required dependencies from `requirements.txt`

3. **Start the Application**
   - Run the Flask server using `python app.py`
   - Access the web interface at `http://localhost:8080`

4. **Verify Installation**
   - Use the health check endpoint to confirm everything is working
   - Test with sample images in the `demo_data/` folder

## 💻 Usage Options

### Web Interface
- **Upload Detection**: Drag and drop images for instant defect analysis
- **Live Camera**: Real-time detection using your webcam
- **Results Visualization**: View annotated images with detected defects
- **Statistics Dashboard**: Monitor detection confidence and counts

### Command Line Tools
- **Batch Processing**: Process multiple images using standalone scripts
- **Live Detection**: Terminal-based real-time detection with statistics
- **Custom Parameters**: Adjust confidence thresholds and camera sources

### REST API Integration
- **Health Monitoring**: Check system and model status
- **Image Prediction**: Send images via Base64 encoding for analysis
- **Model Information**: Retrieve supported defect classes and model details
- **Training Trigger**: Initiate custom model training remotely

## 📊 Detection Results

### Output Information
Each detection provides:
- **Defect Type**: Classification of the detected defect
- **Confidence Score**: Detection reliability (0.0-1.0 scale)
- **Bounding Box**: Precise location coordinates
- **Visual Annotations**: Color-coded boxes on processed images

### Confidence Interpretation
- **High (0.8-1.0)**: Reliable defect detection
- **Medium (0.5-0.8)**: Probable defect presence  
- **Low (0.25-0.5)**: Possible defect requiring review
- **Very Low (<0.25)**: Typically filtered out

## ⚙️ Configuration

### Model Settings
- Default confidence threshold: 0.50 (web interface), 0.25 (live detection)
- IoU threshold: 0.7 for non-maximum suppression
- Model path: `models/best.pt`
- Supported formats: JPG, JPEG, PNG

### Server Configuration
- Host: `0.0.0.0` (all interfaces)
- Port: `8080` (default)
- CORS enabled for cross-origin requests

### Custom Training
- Dataset format: YOLO annotation style
- Supported training parameters: epochs, batch size, image size
- Automatic data validation before training
- Google Drive integration for dataset management

## 📁 Project Structure

```
eaglescan_ai/
│
├── **Web Application Layer**
│   ├── app.py                    # Flask web server & API endpoints
│   ├── templates/
│   │   └── index.html           # Web interface
│   └── static/                  # CSS, JS, images
│
├── **Core ML Package** (`eaglescan/`)
│   ├── components/              # Training pipeline components
│   │   ├── data_ingestion.py   # Dataset download & extraction
│   │   ├── data_validation.py  # Data quality validation
│   │   └── model_trainer.py    # YOLOv11 training logic
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
├── **Standalone Scripts** (`src/`)
│   ├── predict.py              # Batch prediction utility
│   └── live.py                # Real-time detection utility
│
├── **Model & Data**
│   ├── models/                 # Trained model weights
│   │   ├── best.pt            # Best model checkpoint
│   │   └── last.pt            # Latest training checkpoint
│   │
│   ├── demo_data/             # Sample test images
│   ├── config/data.yaml       # Dataset configuration
│   └── evaluation/            # Model evaluation metrics
│
└── **Documentation & Deployment**
    ├── docs/                  # Comprehensive documentation
    ├── research/              # Jupyter notebooks & experiments
    ├── Dockerfile            # Container definition
    ├── requirements.txt      # Python dependencies
    └── setup.py             # Package setup configuration
```

## 🚢 Deployment Options

### Local Development
- Start Flask development server
- Access web interface at `http://localhost:8080`
- Suitable for testing and development

### Docker Deployment
- Build containerized version for consistent deployment
- Supports volume mounting for model weights
- Ideal for production environments

### Cloud Deployment
- Compatible with Azure Container Instances
- Deployable on AWS ECS and Google Cloud Run
- Scalable container orchestration support

## 🔧 Troubleshooting

### Common Issues

**Model Loading Problems**
- Ensure `models/best.pt` file exists in the project directory
- Verify model file permissions and format compatibility
- Check available storage space for model loading

**Camera Access Problems**
- Verify camera permissions in browser and system settings
- Try different camera sources if multiple cameras available
- Ensure camera is not being used by other applications

**Web Interface Issues**
- Confirm Flask server is running and accessible
- Check that port 8080 is not blocked by firewall
- Verify browser compatibility and JavaScript is enabled

**Memory and Performance Issues**
- Close unnecessary applications to free up system memory
- Reduce image sizes for faster processing
- Consider using GPU acceleration if available

**API Connection Problems**
- Verify server address and port configuration
- Check network connectivity and CORS settings
- Validate JSON formatting for API requests

## 📈 Performance Tips

### Optimization Strategies
- **Image Quality**: Use well-lit, high-contrast images for best detection results
- **Processing Speed**: Resize large images before processing to improve speed
- **Accuracy**: Adjust confidence thresholds based on your quality requirements
- **System Resources**: Monitor CPU/GPU usage and memory consumption

### Best Practices
- Maintain consistent lighting conditions during detection
- Keep camera-to-object distance consistent for live detection
- Use perpendicular viewing angles for optimal results
- Ensure images are sharp and properly focused

## 🤝 Contributing

### Development Guidelines
- Follow Python PEP 8 coding standards
- Add comprehensive documentation for new features
- Include unit tests for new functionality
- Update documentation when adding new capabilities

### Adding New Features
- Fork the repository and create feature branches
- Test thoroughly before submitting pull requests
- Update relevant documentation files
- Ensure backward compatibility when possible

## 📚 Documentation

For detailed information, refer to:
- **API Documentation** (`docs/api.md`): Complete API reference and examples
- **Development Guide** (`docs/development.md`): Setup, architecture, and development workflow  
- **User Guide** (`docs/user_guide.md`): Comprehensive usage instructions and troubleshooting

## 🔗 Dependencies

### Core Technologies
- **Python 3.8+**: Programming language foundation
- **YOLOv11 (Ultralytics)**: Object detection model architecture
- **Flask**: Web framework for API and interface
- **OpenCV**: Computer vision processing
- **PyTorch**: Deep learning framework

### Additional Libraries
- **NumPy & SciPy**: Numerical computing
- **Pillow**: Image processing utilities
- **Flask-CORS**: Cross-origin request support
- **gdown**: Google Drive file downloads
- **PyYAML**: Configuration file processing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the YOLOv11 framework
- **Flask** community for web framework
- **OpenCV** for computer vision utilities
- **PyTorch** team for deep learning infrastructure

## 📞 Support & Contact

### Getting Help
- **Documentation**: Check the comprehensive docs in the `docs/` folder
- **GitHub Issues**: Report bugs and request features
- **Community**: Engage with other users and developers

### Project Information
- **Repository**: [GitHub Repository](https://github.com/vany-gr34/starks_v)
- **Version**: Check latest releases for updates and improvements
- **Contributions**: Pull requests and contributions are welcome

---

**EagleScanAI - Enhancing Industrial Quality Control with AI** 🦅
