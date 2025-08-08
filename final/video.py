# Install dependencies
# pip install ultralytics opencv-python matplotlib

from ultralytics import YOLO
import cv2
import time
import os

# Load your trained model locally
model = YOLO(r'C:\Users\vedaa\OneDrive\Desktop\beakyy\final\best.pt')

# Path to the input video
video_path = r"C:\Users\vedaa\OneDrive\Desktop\beakyy\final\multiple-crow.mp4"  # Replace with your video file path

# Output video path
output_path = r"C:\Users\vedaa\OneDrive\Desktop\beakyy\final\output.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process the video
frame_count = 0
start_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform detection on the frame
        results = model(frame)
        
        # Draw the results on the frame
        annotated_frame = results[0].plot()
        
        # Write the frame to the output video
        out.write(annotated_frame)
        
        # Display progress
        frame_count += 1
        if frame_count % 30 == 0:  # Update every 30 frames
            elapsed_time = time.time() - start_time
            frames_per_second = frame_count / elapsed_time
            percentage_complete = (frame_count / total_frames) * 100
            print(f"Progress: {percentage_complete:.1f}% ({frame_count}/{total_frames} frames) at {frames_per_second:.1f} FPS")
            
        # Optionally display the frame (may slow down processing)
        # cv2.imshow('Processing Video', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
            
finally:
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\nVideo processing complete.")
    print(f"Output saved to: {output_path}")
    print(f"Processed {frame_count} frames in {time.time() - start_time:.1f} seconds")