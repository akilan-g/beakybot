import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image, ImageDraw, ImageFont
import time

# Configuration
MODEL_PATH = r'C:\Users\rishi\Desktop\beakyproj\model.tflite'
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['Black_Footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross']

def load_model():
    """Load the TFLite model"""
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def get_input_details(interpreter):
    """Get model input details"""
    return interpreter.get_input_details()

def get_output_details(interpreter):
    """Get model output details"""
    return interpreter.get_output_details()

def preprocess_image(image):
    """Preprocess the image to match model input requirements"""
    # Resize the image
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    image = image.astype(np.float32) / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def predict(interpreter, input_details, output_details, image):
    """Run prediction on an image"""
    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    # Run inference
    interpreter.invoke()
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # Get predicted class and confidence
    class_idx = np.argmax(output_data[0])
    confidence = output_data[0][class_idx]
    return class_idx, confidence

def draw_prediction(frame, class_idx, confidence):
    """Draw prediction on the frame"""
    label = f"{CLASS_NAMES[class_idx]}: {confidence*100:.2f}%"
    
    # Calculate text position (bottom left corner)
    text_x = 10
    text_y = frame.shape[0] - 20
    
    # Draw a semi-transparent background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, text_y - 30), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw the text
    cv2.putText(frame, label, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def main():
    # Load the model
    print("Loading model...")
    interpreter = load_model()
    input_details = get_input_details(interpreter)
    output_details = get_output_details(interpreter)
    print("Model loaded successfully!")

    # Initialize camera
    print("Initializing camera...")
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Camera initialized. Press 'q' to quit.")
    
    # Frame processing variables
    process_this_frame = True
    last_time = time.time()
    fps = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process every other frame to improve performance
            if process_this_frame:
                # Preprocess image
                input_image = preprocess_image(frame)
                
                # Make prediction
                class_idx, confidence = predict(interpreter, input_details, output_details, input_image)
                
                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - last_time)
                last_time = current_time
            
            # Display the prediction on the frame
            result_frame = frame.copy()
            result_frame = draw_prediction(result_frame, class_idx, confidence)
            
            # Display FPS
            cv2.putText(result_frame, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow('Bird Species Detection', result_frame)
            
            # Toggle frame processing
            process_this_frame = not process_this_frame
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Application stopped by user")
    
    finally:
        # Clean up
        camera.release()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    main()