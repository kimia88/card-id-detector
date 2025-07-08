import json
import os

# مسیر فایل JSON اصلی
input_json = 'D:/card-id-detector/my_dataset/instances_default.json'

# مسیر فولدرهای تصاویر train و val
train_images_dir = 'D:/card-id-detector/my_dataset/images/train'
val_images_dir = 'D:/card-id-detector/my_dataset/images/val'

# بارگذاری JSON اصلی
with open(input_json, 'r') as f:
    data = json.load(f)

# گرفتن لیست نام فایل‌های تصویر در train و val
train_files = set(os.listdir(train_images_dir))
val_files = set(os.listdir(val_images_dir))

# تابع برای فیلتر کردن داده‌های JSON
def filter_coco(data, file_names):
    filtered = {}
    filtered['info'] = data.get('info', {})
    filtered['licenses'] = data.get('licenses', [])
    filtered['categories'] = data.get('categories', [])
    
    # تصاویر مربوط به مجموعه مورد نظر
    filtered['images'] = [img for img in data['images'] if img['file_name'] in file_names]
    image_ids = set(img['id'] for img in filtered['images'])
    
    # انوتیشن‌هایی که مربوط به تصاویر انتخاب شده هستن
    filtered['annotations'] = [ann for ann in data['annotations'] if ann['image_id'] in image_ids]
    
    return filtered

# ساخت JSON برای train و val
train_json = filter_coco(data, train_files)
val_json = filter_coco(data, val_files)

# ذخیره فایل‌ها
with open('D:/card-id-detector/my_dataset/train_coco.json', 'w') as f:
    json.dump(train_json, f)

with open('D:/card-id-detector/my_dataset/val_coco.json', 'w') as f:
    json.dump(val_json, f)

print("فایل‌های train و val جدا شدند.")
