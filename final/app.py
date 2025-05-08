from flask import Flask, render_template, request, Response, jsonify
import os
import cv2
import time
import base64
import uuid
from io import BytesIO
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLOv11 model
MODEL_PATH = r"C:\Users\vedaa\OneDrive\Desktop\final-projects\beakyy\final\best.pt"
OUTPUT_DIR = r"C:\Users\vedaa\OneDrive\Desktop\final-projects\beakyy\final\output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global variable for model
model = None

def load_model():
    global model
    if model is None:
        model = YOLO(MODEL_PATH)
    return model

# Instead of using before_first_request, we'll load the model when needed
# or we can use Flask 2.x's new way with app.before_request

# Initialize model at startup - newer Flask way
@app.before_request
def initialize():
    load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Save the uploaded image temporarily
    img_path = os.path.join(OUTPUT_DIR, f"upload_{uuid.uuid4()}.jpg")
    file.save(img_path)
    
    # Load model and run detection
    model = load_model()
    try:
        results = model(img_path, save=True, save_txt=False, project=OUTPUT_DIR)
        
        # Get the result image path
        result_img_path = os.path.join(results[0].save_dir, os.path.basename(img_path))
        
        # Convert result image to base64 for sending to frontend
        with open(result_img_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up temporary files if needed
        if os.path.exists(img_path):
            os.remove(img_path)
        
        # Return detection results
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_data}",
            'detections': len(results[0].boxes),
            'classes': results[0].names
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect_webcam_frame', methods=['POST'])
def detect_webcam_frame():
    try:
        # Get the base64 image from the request
        image_data = request.json.get('image', '')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Remove the data URL prefix to get the base64 string
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode the base64 image
        image_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(image_bytes))
        
        # Convert PIL image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Perform detection
        model = load_model()
        results = model(img_cv)
        
        # Get the annotated image
        annotated_img = results[0].plot()
        
        # Convert back to base64 for sending to frontend
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Return detection results
        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{img_base64}",
            'detections': len(results[0].boxes),
            'classes': results[0].names
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    # Save the uploaded video temporarily
    video_path = os.path.join(OUTPUT_DIR, f"upload_{uuid.uuid4()}.mp4")
    file.save(video_path)
    
    # Process the video with the model
    try:
        output_path = os.path.join(OUTPUT_DIR, f"result_{os.path.basename(video_path)}")
        
        # Load model
        model = load_model()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 500
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform detection
            results = model(frame)
            
            # Draw results on frame
            annotated_frame = results[0].plot()
            
            # Write to output
            out.write(annotated_frame)
        
        # Release resources
        cap.release()
        out.release()
        
        # Return the URL to the processed video
        video_url = f"/get_video/{os.path.basename(output_path)}"
        
        return jsonify({
            'success': True,
            'video_url': video_url
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_video/<filename>')
def get_video(filename):
    video_path = os.path.join(OUTPUT_DIR, filename)
    
    def generate():
        with open(video_path, 'rb') as video_file:
            data = video_file.read(1024)
            while data:
                yield data
                data = video_file.read(1024)
    
    return Response(generate(), mimetype='video/mp4')

# Route to handle livestream detection
@app.route('/start_livestream', methods=['GET'])
def start_livestream():
    return jsonify({'message': 'Livestream functionality is ready'})

@app.route('/templates/index.html')
def serve_index():
    return render_template('index.html')

if __name__ == '__main__':
    # Load model at startup
    model = load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)