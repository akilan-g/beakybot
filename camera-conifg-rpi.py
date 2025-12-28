from picamera2 import Picamera2
from libcamera import controls
import time

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1920, 1080)})
picam2.configure(config)

# 1. Start the camera
picam2.start()

# 2. Set Manual Exposure
# Let's set a 20ms exposure (20,000 microseconds) and a gain of 2.0
picam2.set_controls({
    "AeEnable": False,           # Disable Auto Exposure
    "ExposureTime": 20000,       # 20ms shutter speed
    "AnalogueGain": 2.0,         # Increase sensitivity
    "AwbEnable": False,          # Often helpful to lock White Balance too
    "ColourGains": (1.5, 1.2)    # Manual Red/Blue gains if AWB is off
})

print("Exposure locked. Recording...")

try:
    picam2.start_recording("exposure_test.h264")
    # You can even change exposure while recording!
    time.sleep(5)
    
    print("Brightening image...")
    picam2.set_controls({"ExposureTime": 50000}) # Increase to 50ms
    time.sleep(5)

finally:
    picam2.stop_recording()
    picam2.close()
