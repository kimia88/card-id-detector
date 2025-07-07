from ultralytics import YOLO
import cv2
import os
from pathlib import Path

def run_yolo_detection(input_dir, output_dir, model_path):
    model = YOLO(model_path)  # مدل آموزش‌داده‌شده خودت
    os.makedirs(output_dir, exist_ok=True)

    image_paths = list(Path(input_dir).glob("*.[jp][pn]g")) + list(Path(input_dir).glob("*.jpeg"))

    for img_path in image_paths:
        results = model(str(img_path))
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for i, box in enumerate(results[0].boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            crop = img[y1:y2, x1:x2]
            out_path = Path(output_dir) / f"{Path(img_path).stem}_crop{i}.jpg"
            cv2.imwrite(str(out_path), crop)

    print(f"✅ تشخیص و برش کارت‌ها انجام شد. ({len(image_paths)} عکس پردازش شد)")
