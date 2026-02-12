import sys
import cv2
import os
import json
import time
from pathlib import Path
from datetime import datetime
from eaglescan.pipeline.training_pipeline import TrainPipeline
from eaglescan.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from eaglescan.constant.application import APP_HOST, APP_PORT
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.model_path = "models/best.pt"  
        self.model = None
        self.camera = None
        self.camera_active = False
        self.detection_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'session_start': None,
            'last_detection_time': None,
            'defect_counts': {}
        }
        self.load_model()
    
    def load_model(self):
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
    
    def start_camera(self, camera_id=0):
        """Initialize camera capture"""
        try:
            if self.camera is None or not self.camera.isOpened():
                self.camera = cv2.VideoCapture(camera_id)
                if self.camera.isOpened():
                    self.camera_active = True
                    self.detection_stats['session_start'] = datetime.now().isoformat()
                    logger.info(f"Camera {camera_id} started successfully")
                    return True
                else:
                    logger.error(f"Failed to open camera {camera_id}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            return False
    
    def stop_camera(self):
        """Release camera capture"""
        try:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                self.camera_active = False
                logger.info("Camera stopped successfully")
                return True
        except Exception as e:
            logger.error(f"Error stopping camera: {str(e)}")
            return False
    
    def reset_stats(self):
        """Reset detection statistics"""
        self.detection_stats = {
            'total_frames': 0,
            'total_detections': 0,
            'session_start': datetime.now().isoformat(),
            'last_detection_time': None,
            'defect_counts': {}
        }


@app.route("/train")
def trainRoute():
    try:
        obj = TrainPipeline()
        obj.run_pipeline()
        return jsonify({"message": "Training Successful!!", "status": "success"})
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return jsonify({"message": f"Training failed: {str(e)}", "status": "error"}), 500


@app.route("/")
def home():
    """Render the live camera interface"""
    return render_template("index.html")


@app.route("/upload")
def upload_page():
    """Render the upload interface (if you want to keep both modes)"""
    return render_template("upload.html")


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
        
        # Run prediction
        results = clApp.model.predict(
            source=str(input_path),
            conf=0.80,  
            iou=0.7,   
            save=True,  
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
                    'bbox': box.xyxy[0].tolist()  
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


def generate_frames(conf_threshold=0.70, iou_threshold=0.7):
    """
    Generator function for live camera feed with YOLO detection
    """
    if not clApp.start_camera():
        logger.error("Failed to start camera in generate_frames")
        return
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while clApp.camera_active and clApp.camera is not None:
            success, frame = clApp.camera.read()
            if not success:
                logger.warning("Failed to read frame from camera")
                break

            # Update frame counter
            frame_count += 1
            clApp.detection_stats['total_frames'] += 1

            # Run YOLO prediction on frame
            results = clApp.model.predict(
                source=frame, 
                conf=conf_threshold, 
                iou=iou_threshold,
                verbose=False  # Reduce console output
            )

            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Update detection stats
            detections = results[0].boxes
            if len(detections) > 0:
                clApp.detection_stats['total_detections'] += len(detections)
                clApp.detection_stats['last_detection_time'] = datetime.now().isoformat()
                
                # Count defects by type
                for box in detections:
                    class_name = results[0].names[int(box.cls)]
                    if class_name not in clApp.detection_stats['defect_counts']:
                        clApp.detection_stats['defect_counts'][class_name] = 0
                    clApp.detection_stats['defect_counts'][class_name] += 1

            # Calculate FPS every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                logger.info(f"FPS: {fps:.2f}, Total Detections: {clApp.detection_stats['total_detections']}")
                start_time = time.time()

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                logger.error("Failed to encode frame")
                continue
                
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        logger.error(f"Error in generate_frames: {str(e)}")
    finally:
        clApp.stop_camera()


@app.route("/live")
def live():
    """
    Live camera feed endpoint with real-time YOLO detection
    Query params: conf (confidence threshold), iou (IOU threshold)
    """
    try:
        # Get optional parameters
        conf_threshold = float(request.args.get('conf', 0.70))
        iou_threshold = float(request.args.get('iou', 0.7))
        
        return Response(
            generate_frames(conf_threshold, iou_threshold),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"Error starting live feed: {str(e)}")
        return jsonify({"message": f"Failed to start live feed: {str(e)}", "status": "error"}), 500


@app.route("/camera/start", methods=['POST'])
@cross_origin()
def start_camera():
    """
    Start the camera
    Optional JSON: {"camera_id": 0}
    """
    try:
        camera_id = request.json.get('camera_id', 0) if request.json else 0
        
        if clApp.start_camera(camera_id):
            return jsonify({
                "message": "Camera started successfully",
                "status": "success",
                "camera_active": True
            })
        else:
            return jsonify({
                "message": "Failed to start camera",
                "status": "error",
                "camera_active": False
            }), 500
    except Exception as e:
        logger.error(f"Error in start_camera endpoint: {str(e)}")
        return jsonify({"message": str(e), "status": "error"}), 500


@app.route("/camera/stop", methods=['POST'])
@cross_origin()
def stop_camera():
    """Stop the camera"""
    try:
        if clApp.stop_camera():
            return jsonify({
                "message": "Camera stopped successfully",
                "status": "success",
                "camera_active": False
            })
        else:
            return jsonify({
                "message": "Failed to stop camera",
                "status": "error"
            }), 500
    except Exception as e:
        logger.error(f"Error in stop_camera endpoint: {str(e)}")
        return jsonify({"message": str(e), "status": "error"}), 500


@app.route("/camera/status", methods=['GET'])
def camera_status():
    """Get current camera status"""
    try:
        return jsonify({
            "camera_active": clApp.camera_active,
            "camera_available": clApp.camera is not None and clApp.camera.isOpened(),
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error getting camera status: {str(e)}")
        return jsonify({"message": str(e), "status": "error"}), 500


@app.route("/stats", methods=['GET'])
def get_stats():
    """
    Get current detection statistics
    Returns session stats including frame count, detections, etc.
    """
    try:
        stats = clApp.detection_stats.copy()
        
        # Calculate session duration
        if stats['session_start']:
            session_start = datetime.fromisoformat(stats['session_start'])
            duration = (datetime.now() - session_start).total_seconds()
            stats['session_duration_seconds'] = duration
        else:
            stats['session_duration_seconds'] = 0
        
        return jsonify({
            "stats": stats,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return jsonify({"message": str(e), "status": "error"}), 500


@app.route("/stats/reset", methods=['POST'])
@cross_origin()
def reset_stats():
    """Reset detection statistics"""
    try:
        clApp.reset_stats()
        return jsonify({
            "message": "Statistics reset successfully",
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error resetting stats: {str(e)}")
        return jsonify({"message": str(e), "status": "error"}), 500


@app.route("/snapshot", methods=['POST'])
@cross_origin()
def take_snapshot():
    """
    Capture a single frame from the camera
    Returns base64 encoded image with detections
    """
    try:
        if not clApp.camera_active or clApp.camera is None:
            return jsonify({
                "message": "Camera is not active",
                "status": "error"
            }), 400
        
        # Capture frame
        success, frame = clApp.camera.read()
        if not success:
            return jsonify({
                "message": "Failed to capture frame",
                "status": "error"
            }), 500
        
        # Run detection
        conf_threshold = request.json.get('conf', 0.75) if request.json else 0.75
        results = clApp.model.predict(source=frame, conf=conf_threshold, iou=0.7)
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        # Extract detections
        detections = []
        for box in results[0].boxes:
            detection = {
                'class': results[0].names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': box.xyxy[0].tolist()
            }
            detections.append(detection)
        
        # Save snapshot
        snapshot_dir = Path("snapshots")
        snapshot_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = snapshot_dir / f"snapshot_{timestamp}.jpg"
        cv2.imwrite(str(snapshot_path), annotated_frame)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = encodeImageIntoBase64(str(snapshot_path))
        
        return jsonify({
            "image": img_base64.decode('utf-8'),
            "detections": detections,
            "count": len(detections),
            "timestamp": timestamp,
            "path": str(snapshot_path),
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Error taking snapshot: {str(e)}")
        return jsonify({"message": str(e), "status": "error"}), 500


@app.route("/health", methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": clApp.model is not None,
        "model_path": clApp.model_path,
        "camera_active": clApp.camera_active
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
                "model_path": clApp.model_path,
                "status": "loaded"
            })
        else:
            return jsonify({"message": "Model not loaded", "status": "error"}), 500
    except Exception as e:
        return jsonify({"message": str(e), "status": "error"}), 500


@app.route("/config", methods=['GET', 'POST'])
@cross_origin()
def config():
    """
    GET: Retrieve current configuration
    POST: Update configuration
    """
    if request.method == 'GET':
        return jsonify({
            "confidence_threshold": 0.75,
            "iou_threshold": 0.7,
            "camera_id": 0,
            "status": "success"
        })
    else:
        try:
            # Update configuration (you can store these in class variables)
            config_data = request.json
            return jsonify({
                "message": "Configuration updated",
                "config": config_data,
                "status": "success"
            })
        except Exception as e:
            return jsonify({"message": str(e), "status": "error"}), 500


if __name__ == "__main__":
    # Initialize the client app and load model
    clApp = ClientApp()
    
    # Run Flask app
    logger.info(f"Starting EagleScanAI on {APP_HOST}:{APP_PORT}")
    logger.info("Endpoints available:")
    logger.info("  - GET  /               : Live camera interface")
    logger.info("  - GET  /upload         : Upload interface")
    logger.info("  - GET  /live           : Live camera stream")
    logger.info("  - POST /camera/start   : Start camera")
    logger.info("  - POST /camera/stop    : Stop camera")
    logger.info("  - GET  /camera/status  : Camera status")
    logger.info("  - GET  /stats          : Detection statistics")
    logger.info("  - POST /stats/reset    : Reset statistics")
    logger.info("  - POST /snapshot       : Capture snapshot")
    logger.info("  - POST /predict        : Predict from image")
    logger.info("  - GET  /health         : Health check")
    logger.info("  - GET  /model-info     : Model information")
    
    app.run(host=APP_HOST, port=APP_PORT, debug=True, threaded=True)