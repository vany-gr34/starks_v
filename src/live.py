from ultralytics import YOLO
import cv2
import time

def live_detection(model_path='models/best.pt', confidence=0.50):
    """
    Run live defect detection with terminal statistics
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    print(f"Model loaded! Classes: {list(model.names.values())}")
    print("\nStarting live detection...")
    print("Press 'q' to quit\n")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break
            
            # Run detection
            results = model.predict(
                source=frame,
                conf=confidence,
                verbose=False  # Suppress per-frame output
            )
            
            # Get annotated frame
            annotated_frame = results[0].plot()
            
            # Extract detections
            detections = results[0].boxes
            frame_count += 1
            
            # Print detection info in terminal
            if len(detections) > 0:
                print(f"\n--- Frame {frame_count} ---")
                for box in detections:
                    class_name = model.names[int(box.cls)]
                    conf = float(box.conf)
                    print(f"  ✓ Detected: {class_name} (confidence: {conf:.2f})")
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add FPS to frame
            cv2.putText(
                annotated_frame, 
                f"FPS: {fps:.1f} | Detections: {len(detections)}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Show frame
            cv2.imshow('Defect Detection - Press Q to quit', annotated_frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\nStopping detection...")
                break
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user...")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Session Summary:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Average FPS: {frame_count/elapsed:.1f}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Live defect detection')
    parser.add_argument('--model', type=str, default='../weights/best.pt', 
                        help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.50, 
                        help='Confidence threshold')
    parser.add_argument('--source', type=int, default=0, 
                        help='Camera source (0=default, 1=external)')
    
    args = parser.parse_args()
    
    live_detection(model_path=args.model, confidence=args.conf)