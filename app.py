import sys
import os
from pathlib import Path
from starks.pipeline.training_pipeline import TrainPipeline
from starks.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from starks.constant.application import APP_HOST, APP_PORT
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.model_path = "weights/best.pt"  
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the YOLO11 model at startup"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"Model loaded successfully from {self.model_path}")
            else:
                logger.error(f"Model not found at {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise


@app.route("/train")
def trainRoute():
    """Trigger training pipeline"""
    try:
        obj = TrainPipeline()
        obj.run_pipeline()
        return jsonify({"message": "Training Successful!!", "status": "success"})
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return jsonify({"message": f"Training failed: {str(e)}", "status": "error"}), 500


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    """
    Predict defects from uploaded image
    Expects JSON: {"image": "base64_encoded_image"}
    Returns JSON: {"image": "base64_annotated_image", "detections": [...]}
    """
    try:
        # Decode the incoming image
        image = request.json['image']
        input_path = Path("data") / clApp.filename
        input_path.parent.mkdir(parents=True, exist_ok=True)
        decodeImage(image, str(input_path))
        
        logger.info(f"Processing image: {input_path}")
        
        # Run YOLO11 prediction
        results = clApp.model.predict(
            source=str(input_path),
            conf=0.25,  # Confidence threshold
            iou=0.7,    # IoU threshold for NMS
            save=True,  # Save annotated image
            project='runs/detect',
            name='exp',
            exist_ok=True
        )
        
        # Extract detection information
        detections = []
        for r in results:
            for box in r.boxes:
                detection = {
                    'class': r.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        logger.info(f"Found {len(detections)} defects")
        
        # Encode the annotated image back to base64
        output_image_path = Path("runs/detect/exp") / clApp.filename
        
        if output_image_path.exists():
            opencodedbase64 = encodeImageIntoBase64(str(output_image_path))
            result = {
                "image": opencodedbase64.decode('utf-8'),
                "detections": detections,
                "count": len(detections),
                "status": "success"
            }
        else:
            result = {
                "message": "Prediction completed but output image not found",
                "detections": detections,
                "count": len(detections),
                "status": "warning"
            }
        
        # Cleanup
        if Path("runs/detect").exists():
            import shutil
            shutil.rmtree("runs/detect")
        
        return jsonify(result)

    except ValueError as val:
        logger.error(f"ValueError: {val}")
        return jsonify({"message": "Value not found inside json data", "status": "error"}), 400
    
    except KeyError as e:
        logger.error(f"KeyError: {e}")
        return jsonify({"message": f"Missing key: {str(e)}", "status": "error"}), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"message": f"Prediction failed: {str(e)}", "status": "error"}), 500


@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    """
    Start live camera detection
    Uses webcam (source=0) for real-time defect detection
    """
    try:
        logger.info("Starting live camera detection...")
        
        # Run live detection on webcam
        clApp.model.predict(
            source=0,  # Webcam
            conf=0.25,
            iou=0.7,
            show=True,  # Display results in window
            save=False,
            stream=True  # Enable streaming mode
        )
        
        return jsonify({"message": "Camera started successfully!", "status": "success"})
    
    except ValueError as val:
        logger.error(f"ValueError: {val}")
        return jsonify({"message": str(val), "status": "error"}), 400
    
    except Exception as e:
        logger.error(f"Live detection error: {str(e)}")
        return jsonify({"message": f"Camera failed: {str(e)}", "status": "error"}), 500


@app.route("/health", methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": clApp.model is not None,
        "model_path": clApp.model_path
    })


@app.route("/model-info", methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if clApp.model:
            return jsonify({
                "model_type": "YOLO11n",
                "classes": list(clApp.model.names.values()),
                "num_classes": len(clApp.model.names),
                "status": "loaded"
            })
        else:
            return jsonify({"message": "Model not loaded", "status": "error"}), 500
    except Exception as e:
        return jsonify({"message": str(e), "status": "error"}), 500


if __name__ == "__main__":
    # Initialize the client app and load model
    clApp = ClientApp()
    
    # Run Flask app
    logger.info(f"Starting Flask app on {APP_HOST}:{APP_PORT}")
    app.run(host=APP_HOST, port=APP_PORT, debug=True)