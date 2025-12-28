from picamera2 import Picamera2
import time

# Initialize camera
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1920, 1080)})
picam2.configure(config)

# Enable Continuous Autofocus for Module 3
picam2.set_controls({"AfMode": 2})

output_file = "manual_record.h264"

try:
    picam2.start_preview()
    print(f"Recording to {output_file}...")
    print("Press Ctrl+C to stop recording.")
    
    picam2.start_recording(output_file)
    
    # Keep the script running until the user interrupts
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nStop command received.")

finally:
    # This block ensures the camera stops and saves the file correctly
    picam2.stop_recording()
    picam2.close()
    print("Recording saved and camera released.")
