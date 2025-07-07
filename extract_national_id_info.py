import os
import json

def convert(size, box):
    """
    Converts COCO bbox to YOLO format.
    box = [x_min, y_min, width, height]
    size = (image_width, image_height)
    returns: [x_center, y_center, width, height] normalized
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    return (x * dw, y * dh, w * dw, h * dh)

def run_conversion():
    # مسیر فایل COCO
    coco_json_path = "D:/card-id-detector/annotations/instances_default.json"

    # مسیر خروجی فایل‌های YOLO .txt
    output_dir = "D:/card-id-detector/labels_yolo"
    os.makedirs(output_dir, exist_ok=True)

    # خواندن فایل JSON
    with open(coco_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # نگاشت image_id به اطلاعات عکس
    image_id_to_info = {img["id"]: img for img in data["images"]}

    # تبدیل annotationها به فایل‌های YOLO txt
    for ann in data["annotations"]:
        image_info = image_id_to_info[ann["image_id"]]
        width = image_info["width"]
        height = image_info["height"]
        bbox = ann["bbox"]
        yolo_box = convert((width, height), bbox)
        class_id = ann["category_id"] - 1  # اگر YOLO از صفر شروع می‌کنه

        file_name = os.path.splitext(image_info["file_name"])[0] + ".txt"
        label_path = os.path.join(output_dir, file_name)

        with open(label_path, 'a') as f:
            f.write(f"{class_id} " + " ".join([f"{a:.6f}" for a in yolo_box]) + "\n")

    print(f"✅ تبدیل انجام شد! تعداد فایل برچسب: {len(os.listdir(output_dir))}")

if __name__ == "__main__":
    run_conversion()
