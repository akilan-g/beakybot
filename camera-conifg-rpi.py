from picamera2 import Picamera2
import time

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1920, 1080)})
picam2.configure(config)
picam2.start()

# Exposure settings
picam2.set_controls({"AeEnable": False, "ExposureTime": 20000, "AnalogueGain": 2.0})

print("Exposure locked. Recording...")

try:
    # Explicitly define the output and encoder for this version of Picamera2
    picam2.start_recording(output="exposure_test.h264")
    
    input("Recording... Press ENTER to stop.")

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    picam2.stop_recording()
    picam2.close()
    print("Camera closed.")
