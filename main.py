import os
import shutil
from yolo_detection import run_yolo_detection
from batch_rotate_from_multiple_folders import run_rotation
from extract_text_with_bbox import run_ocr_extraction

def clean_folder(folder_path):
    """اگر پوشه وجود دارد، حذف و دوباره ایجاد می‌کند"""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

def main():
    # مسیرها
    input_folder = "input"                 # عکس‌های کامل ورودی
    cropped_folder = "cropped_cards"      # کارت‌های برش‌خورده توسط YOLO
    rotated_folder = "rotated_output"     # کارت‌های چرخیده شده
    ocr_output_folder = "ocr_outputs_pos" # خروجی نهایی OCR
    yolo_model_path = "runs/detect/train2/weights/best.pt"  # مسیر مدل YOLO آموزش‌داده‌شده

    # پاک‌سازی خروجی‌های قبلی
    clean_folder(cropped_folder)
    clean_folder(rotated_folder)
    clean_folder(ocr_output_folder)

    print("شروع تشخیص و برش کارت‌ها با YOLO ...")
    run_yolo_detection(input_folder, cropped_folder, yolo_model_path)

    print("شروع چرخش هوشمند کارت‌ها ...")
    run_rotation([cropped_folder], rotated_folder)

    print("شروع استخراج متن و اطلاعات کارت‌ها ...")
    run_ocr_extraction(rotated_folder, ocr_output_folder)

    print("🎉 همه مراحل با موفقیت انجام شد.")

if __name__ == "__main__":
    main()
