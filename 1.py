import json
import os

input_json = 'D:/card-id-detector/my_dataset/instances_default.json'

labels_dir = 'D:/card-id-detector/my_dataset/labels'
images_dir = 'D:/card-id-detector/my_dataset/images'

# بارگذاری JSON
with open(input_json, 'r') as f:
    data = json.load(f)

# ساخت دیکشنری از تصاویر بر اساس id
images = {img['id']: img for img in data['images']}

# کلاس‌ها (اینجا فقط 1 کلاس به اسم CARD هست)
categories = {cat['id']: idx for idx, cat in enumerate(data['categories'])}

# تابع تبدیل bbox COCO به فرمت YOLO
def bbox_coco_to_yolo(box, width, height):
    x, y, w, h = box
    x_center = x + w/2
    y_center = y + h/2
    return x_center/width, y_center/height, w/width, h/height

# ساخت پوشه‌های labels/train و labels/val اگر وجود ندارند
os.makedirs(os.path.join(labels_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(labels_dir, 'val'), exist_ok=True)

# تقسیم تصاویر به train و val بر اساس نام فایل در پوشه images
train_files = set(os.listdir(os.path.join(images_dir, 'train')))
val_files = set(os.listdir(os.path.join(images_dir, 'val')))

for ann in data['annotations']:
    image_info = images[ann['image_id']]
    file_name = image_info['file_name']
    
    # تعیین پوشه خروجی براساس اینکه تصویر در train هست یا val
    if file_name in train_files:
        label_subdir = 'train'
    elif file_name in val_files:
        label_subdir = 'val'
    else:
        continue  # اگر تصویر تو train یا val نیست، رد می‌کنیم
    
    width = image_info['width']
    height = image_info['height']
    
    # تبدیل bbox به فرمت YOLO
    bbox = bbox_coco_to_yolo(ann['bbox'], width, height)
    
    class_id = categories[ann['category_id']]
    
    # نام فایل txt برای لیبل
    txt_file = os.path.join(labels_dir, label_subdir, file_name.replace('.jpg', '.txt'))
    
    # نوشتن خطوط (چندین شی ممکن است در یک تصویر باشد)
    with open(txt_file, 'a') as f:
        f.write(f"{class_id} {' '.join(map(str, bbox))}\n")

print("تبدیل JSON COCO به فایل‌های YOLO txt انجام شد.")
