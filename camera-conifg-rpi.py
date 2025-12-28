from picamera2 import Picamera2
from picamera2.outputs import FileOutput # Import the output helper
import time

picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1920, 1080)})
picam2.configure(config)

picam2.start()

# Lock exposure settings
picam2.set_controls({
    "AeEnable": False,
    "ExposureTime": 20000,
    "AnalogueGain": 2.0
})

print("Exposure locked. Recording...")

try:
    # USE THIS SYNTAX for Picamera2
    picam2.start_recording(FileOutput("exposure_test.h264"))
    
    time.sleep(10)

finally:
    picam2.stop_recording()
    picam2.close()
