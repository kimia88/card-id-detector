from ultralytics import YOLO

# 🧠 مدل آموزش‌دیده‌ات رو اینجا بذار
model = YOLO("best.pt")  # اگر هنوز نداری، باید آموزش بدی

# 🖼️ عکس ورودی
image_path = "test.jpg"

# 🔍 اجرای تشخیص
results = model(image_path, conf=0.4, save=True)

# 💬 نمایش نتایج در ترمینال
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"کارت ملی پیدا شد - اعتماد: {conf:.2f}")
