from ultralytics import YOLO

model = YOLO('yolov8m.pt')  # مدل پیش‌آموزش‌دیده متوسط

model.train(data='data.yaml', epochs=150, imgsz=640, batch=16, patience=30)
