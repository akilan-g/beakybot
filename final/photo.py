from ultralytics import YOLO
import os

def detect_with_yolov11(weights_path, image_path, output_dir=r"C:\Users\vedaa\OneDrive\Desktop\beakyy\final\output"):
    # Load YOLOv11 model
    model = YOLO(weights_path)

    # Run detection
    results = model(image_path, save=True, save_txt=False, project=output_dir)

    # Get output path
    output_image_path = os.path.join(results[0].save_dir, os.path.basename(image_path))
    print(f"[✓] Detection done! Saved to: {output_image_path}")
    return output_image_path

# === Example Usage ===
weights_path = r"C:\Users\vedaa\OneDrive\Desktop\beakyy\final\best.pt"
image_path = r"C:\Users\vedaa\OneDrive\Desktop\beakyy\final\peacock.webp"
detect_with_yolov11(weights_path, image_path)