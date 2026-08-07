from ultralytics import YOLO
import cv2
import numpy as np

class RealTimeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)  # Webcam
        self.cap.set(3, 640)  # Width
        self.cap.set(4, 640) #Height

    def generate_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break
            
            # Perform detection
            results = self.model(frame)
            annotated_frame = results[0].plot()
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def release(self):
        self.cap.release()