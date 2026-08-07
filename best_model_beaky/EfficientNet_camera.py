import cv2
import numpy as np
import tensorflow as tf
import time

# Bird species names (based on your folder structure)
BIRD_CLASSES = [
    'Asian Green Bee-Eater', 'Brown-Headed Barbet', 'Cattle Egret', 
    'Common Kingfisher', 'Common Myna', 'Common Rosefinch',
    'Common Tailorbird', 'Coppersmith Barbet', 'Forest Wagtail',
    'Grey Wagtail', 'Hoopoe', 'House Crow', 'Indian Grey Hornbill',
    'Indian Peacock', 'Indian Pitta', 'Indian Roller', 'Jungle Babbler',
    'Northern Lapwing', 'Red-Wattled Lapwing', 'Rufous Treepie',
    'Ruddy Shelduck', 'Sarus Crane', 'White Wagtail',
    'White-Breasted Kingfisher', 'White-Breasted Waterhen'
]

def load_tflite_model(model_path):
    """Load TFLite model and allocate tensors."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_model_details(interpreter):
    """Get input and output details from the model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
    
    # Print model details for debugging
    print(f"Model Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
    print(f"Model Output: {output_details[0]['shape']} {output_details[0]['dtype']}")
    
    return input_details, output_details, input_height, input_width

def preprocess_image(image, input_height, input_width):
    """Enhanced preprocessing for better detection of birds on phone screens."""
    # First, apply some image enhancements to reduce screen artifacts
    # 1. Reduce glare/reflections with adaptive histogram equalization
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 2. Apply slight sharpening to improve edge detection
    kernel = np.array([[-1, -1, -1], 
                       [-1,  9, -1], 
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced_img, -1, kernel)
    
    # 3. Reduce noise with a slight blur
    processed_img = cv2.GaussianBlur(sharpened, (3, 3), 0)
    
    # 4. Resize the image to match the input shape
    resized_img = cv2.resize(processed_img, (input_width, input_height))
    
    # 5. Convert to RGB (OpenCV uses BGR by default)
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    
    # 6. Normalize pixel values
    # Using same normalization as EfficientNet: rescale to [-1,1]
    normalized_img = (rgb_img.astype(np.float32) / 127.5) - 1.0
    
    # Alternative normalization: [0,1] if your model was trained this way
    # normalized_img = rgb_img.astype(np.float32) / 255.0
    
    # 7. Add batch dimension
    input_tensor = np.expand_dims(normalized_img, axis=0)
    
    return input_tensor, resized_img

def predict_bird(interpreter, image, input_details, output_details):
    """Predict bird species using TFLite model with detailed output."""
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Get all class probabilities for debugging
    all_scores = output_data[0]
    
    # Get top 3 predictions for display
    top_indices = np.argsort(all_scores)[-3:][::-1]
    top_predictions = [(BIRD_CLASSES[i], float(all_scores[i])) for i in top_indices]
    
    return top_predictions

def debug_prediction(frame, top_predictions):
    """Show more detailed prediction information for debugging."""
    # Create a black rectangle at the bottom for predictions
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h-120), (w, h), (0, 0, 0), -1)
    
    # Display top 3 predictions with probabilities
    for i, (bird, prob) in enumerate(top_predictions):
        color = (0, 255, 0) if i == 0 else (200, 200, 200)
        y_pos = h - 90 + (i * 30)
        cv2.putText(frame, f"{i+1}. {bird}: {prob:.4f}", 
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame

def main():
    # Path to your TFLite model
    model_path = r"C:\Users\rishi\Desktop\best_model_beaky\final_model.tflite"  # Replace with your actual model path
    
    # Load TFLite model
    try:
        interpreter = load_tflite_model(model_path)
        input_details, output_details, input_height, input_width = get_model_details(interpreter)
        print(f"Model loaded successfully. Input shape: {input_height}x{input_width}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 for default camera
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Set frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Add a processing mode toggle
    processing_modes = ["Enhanced", "Standard", "Raw"]
    current_mode = 0
    
    # Control variables
    frame_count = 0
    last_prediction_time = time.time()
    prediction_interval = 0.2  # Make predictions more frequently (5 per second)
    
    # Create named window and add trackbar for confidence threshold
    cv2.namedWindow('Bird Species Detection')
    confidence_threshold = 15  # Start with a lower threshold (0.15)
    cv2.createTrackbar('Confidence %', 'Bird Species Detection', confidence_threshold, 100, lambda x: None)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Get current confidence threshold from trackbar (divide by 100 to get 0.0-1.0 range)
        confidence_threshold = cv2.getTrackbarPos('Confidence %', 'Bird Species Detection') / 100.0
        
        current_time = time.time()
        display_frame = frame.copy()
        
        # Make prediction every prediction_interval seconds
        if current_time - last_prediction_time >= prediction_interval:
            # Preprocess the image based on current mode
            if processing_modes[current_mode] == "Raw":
                # Minimal preprocessing
                processed_image = np.expand_dims(
                    cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                              (input_width, input_height)).astype(np.float32) / 255.0, 
                    axis=0)
                debug_img = cv2.resize(frame, (input_width, input_height))
            else:
                # Enhanced preprocessing
                processed_image, debug_img = preprocess_image(frame, input_height, input_width)
            
            # Predict bird species
            top_predictions = predict_bird(interpreter, processed_image, input_details, output_details)
            
            last_prediction_time = current_time
        
        # Show processing mode
        cv2.putText(display_frame, f"Mode: {processing_modes[current_mode]}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show threshold
        cv2.putText(display_frame, f"Threshold: {confidence_threshold:.2f}", 
                   (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display predictions with enhanced debugging
        display_frame = debug_prediction(display_frame, top_predictions)
        
        # Display the processed (small) image in a corner for reference
        if frame_count % 30 == 0:  # Update debug image every 30 frames
            h, w = display_frame.shape[:2]
            debug_h, debug_w = debug_img.shape[:2]
            scale = 200 / debug_h
            debug_display = cv2.resize(debug_img, (int(debug_w * scale), int(debug_h * scale)))
            display_frame[70:70+debug_display.shape[0], 10:10+debug_display.shape[1]] = debug_display
        
        # Display frame
        cv2.imshow('Bird Species Detection', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            # Toggle processing mode
            current_mode = (current_mode + 1) % len(processing_modes)
            print(f"Switched to {processing_modes[current_mode]} mode")
        
        frame_count += 1
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()