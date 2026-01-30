from ultralytics import YOLO
import cv2
import os

class DefectDetector:
    def __init__(self, model_path='../weights/best.pt'):
        
        self.model = YOLO(model_path)
        
    def predict_image(self, image_path, conf_threshold=0.25):
        """Run detection on a single image"""
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            save=True,  # Save annotated image
            project='runs/detect',
            name='predict'
        )
        
        # Extract detections
        detections = []
        for r in results:
            for box in r.boxes:
                detection = {
                    'class': r.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        return detections
    
    def predict_folder(self, folder_path, conf_threshold=0.25):
        """Run detection on all images in a folder"""
        results = self.model.predict(
            source=folder_path,
            conf=conf_threshold,
            save=True,
            project='runs/detect',
            name='batch_predict'
        )
        return results

# Usage example
if __name__ == "__main__":
    # Initialize detector
    detector = DefectDetector()
    
    # Test on single image
    image_path = '../test_images/testo.jpeg'
    detections = detector.predict_image(image_path, conf_threshold=0.3)
    
    
    print(f"Found {len(detections)} defects:")
    for det in detections:
        print(f"- {det['class']}: {det['confidence']:.2f}")