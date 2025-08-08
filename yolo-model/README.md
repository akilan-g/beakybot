# 🧠 YOLOv11 NCNN Object Detection on Raspberry Pi 5

**real-time bird detection** using **YOLOv11 models** in the **NCNN format**, optimized for edge devices like the **Raspberry Pi 5**.

---


## 📦 Prerequisites

### 🔧 System Setup

Update and install system-level packages:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install git python3 python3-pip python3-opencv libopencv-dev wget
sudo apt install python3-picamera2 libcamera-apps  # (For Pi Camera)
```

---

## 📥 Install Dependencies

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧪 Verify System Compatibility

Ensure the following software is working:

- `python3 --version`
- `pip --version`
- `libcamera-hello` (for Pi camera test)
- `opencv_version` (check OpenCV installation)

---
## 🔄 Convert YOLOv11 `.pt` to NCNN Format

First, install the required export tools:

```bash
pip install ultralytics ncnn
```

Then, use Ultralytics to export your trained model to NCNN format:

```bash
yolo export model=best.pt format=ncnn
```

This will generate two files:

- `best_ncnn_model.param`
- `best_ncnn_model.bin`

✅ **Make sure both files are present in your working directory before proceeding.**


## 📸 Run Live Object Detection

Use the Pi Camera or USB webcam to start detection:

```bash
python3 yolo_detect.py --model=best_ncnn_model --source=picamera0 --resolution=640x480
```

For USB webcam, use:

```bash
python3 yolo_detect.py --model=best_ncnn_model --source=usb0 --resolution=640x480
```

---

## 🧪 Camera Troubleshooting

To test Pi Camera module:

```bash
rpicam-hello
```

Ensure:

- Camera ribbon is connected properly
- Camera interface is enabled in `raspi-config`
- Required libraries like `libcamera` are installed

---

## 🛠️ Customization (Optional)

You can modify `yolo_detect.py` to:

- Save annotated images or videos
- Log detection results (e.g., timestamp, class, confidence)
- Customize bounding box colors, font, thickness

---

## 📊 Performance Tip

Use smaller models like `YOLOv11n` or `YOLOv11s` and a resolution of `640x480` for best performance on the Raspberry Pi 5.

![YOLO format speed](./yolo-format-speeds.png)

✅ **NCNN delivers the fastest inference and lowest memory usage** among PyTorch, ONNX, and TFLite formats — perfect for edge AI.

---

## 📄 `requirements.txt`

Save the provided list of packages as `requirements.txt`.  
Install with:

```bash
pip install -r requirements.txt
```

<details>
<summary>📘 Click to view full <code>requirements.txt</code></summary>

```txt
asgiref==3.6.0
astroid==2.14.2
...
ultralytics==8.3.155
ultralytics-thop==2.0.14
...
wrapt==1.14.1
zipp==1.0.0
```

</details>

---

## ✅ Checklist

- [x] Install system dependencies ✅  
- [x] Install Python requirements ✅  
- [x] Convert `best.pt` to NCNN ✅  
- [x] Verify model files exist ✅  
- [x] Run detection using Pi or USB camera ✅  

# Run command example
```bash

python3 yolo_detect.py --model=best_ncnn_model --source=picamera0 --resolution=640x480
```

---
