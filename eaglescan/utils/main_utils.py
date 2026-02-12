import os.path
import sys
import yaml
import base64
from pathlib import Path

from eaglescan.exception import AppException
from eaglescan.logger import logging
import cv2

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            logging.info("Read yaml file successfully")
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise AppException(e, sys) from e
    



def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)
            logging.info("Successfully write_yaml_file")

    except Exception as e:
        raise AppException(e, sys)
    



def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    file_path = Path(fileName)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(fileName, 'wb') as f:
        f.write(imgdata)
    
    logging.info(f"Image decoded and saved to: {fileName}")

def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())



def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO prediction on frame
        results = clApp.model.predict(source=frame, conf=0.25, iou=0.7)

        annotated_frame = results[0].plot()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

    