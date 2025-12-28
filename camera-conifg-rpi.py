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
    # Use the filename string directly here; 
    # newer Picamera2 versions handle the FileOutput wrapper automatically 
    # if you don't pass a custom encoder.
    picam2.start_recording("exposure_test.h264")
    
    # Alternatively, if that STILL fails, use the keyword explicitly:
    # picam2.start_recording(output="exposure_test.h264")
    
    time.sleep(10)

finally:
    picam2.stop_recording()
    picam2.close()
