import os
import shutil

# تنظیم مسیرها
images_base = "images"
labels_base = "labels"
labels_all_dir = "labels_yolo"

# ساختن پوشه‌ها اگر وجود ندارند
os.makedirs(os.path.join(labels_base, "train"), exist_ok=True)
os.makedirs(os.path.join(labels_base, "val"), exist_ok=True)

def move_labels(images_dir, labels_dest_dir):
    moved = 0
    for img_file in os.listdir(images_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(img_file)[0]
            label_file = f"{base_name}.txt"
            label_path = os.path.join(labels_all_dir, label_file)
            dest_path = os.path.join(labels_dest_dir, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path, dest_path)
                moved += 1
            else:
                print(f"⚠️ برچسب پیدا نشد: {label_file}")
    print(f"\n📁 از {images_dir} → {labels_dest_dir}: {moved} فایل برچسب منتقل شد.")

# اجرای تابع برای هر مجموعه
move_labels(os.path.join(images_base, "train"), os.path.join(labels_base, "train"))
move_labels(os.path.join(images_base, "val"), os.path.join(labels_base, "val"))
