from ultralytics import YOLO

# 1. Load the lightweight pre-trained AI model (it auto-downloads 'yolov8n.pt' on first run, ~6 MB)
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')

# 2. Run detection on a sample image hosted online
# It downloads the image, runs inference, and saves the annotated result
print("Running AI detection...")
results = model.predict(source='https://ultralytics.com/images/bus.jpg', save=True)

# 3. Print what the AI found
print("\n--- Detection Complete! ---")
for result in results:
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        print(f"Detected: {class_name.upper()} ({confidence * 100:.1f}% confidence)")

print("\nThe processed image with bounding boxes is saved inside: runs/detect/predict/")