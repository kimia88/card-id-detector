from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train/weights/best.pt')  # مسیر وزن بهترین مدل بعد از آموزش

results = model.predict(source='D:/card-id-detector/my_dataset/images/train/1.jpg', conf=0.25)

# نمایش تصویر با باکس
for result in results:
    img = result.plot()  # تصویر با جعبه‌های تشخیص داده شده
    cv2.imshow('Detected', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
