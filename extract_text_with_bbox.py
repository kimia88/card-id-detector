import os
import json
import easyocr

input_folder = r"D:\card-id-detector\rotated_output"
output_folder = r"D:\card-id-detector\ocr_outputs_pos"
os.makedirs(output_folder, exist_ok=True)

reader = easyocr.Reader(['fa'])

def fix_text(text):
    return text.replace('ي', 'ی').replace('ك', 'ک').strip()

def categorize_texts(results, height):
    national_id = None
    name = None
    family = None
    father = None
    birth = None
    expire = None

    for (text, box) in results:
        text_fixed = fix_text(text)
        y_center = sum([p[1] for p in box]) / 4
        x_center = sum([p[0] for p in box]) / 4

        if len(text_fixed) == 10 and text_fixed.isdigit():
            national_id = text_fixed

        elif 'تولد' in text_fixed or '۱۳' in text_fixed:
            birth = text_fixed

        elif 'اعتبار' in text_fixed:
            expire = text_fixed

        elif y_center < height * 0.4:
            if 'پدر' in text_fixed:
                father = text_fixed
        elif y_center < height * 0.75:
            if name is None and 'نام' in text_fixed and 'خانوادگی' not in text_fixed:
                name = text_fixed
            elif family is None and 'نام خانوادگی' in text_fixed:
                family = text_fixed

    return {
        'شماره ملی': national_id,
        'نام': name,
        'نام خانوادگی': family,
        'نام پدر': father,
        'تاریخ تولد': birth,
        'پایان اعتبار': expire
    }

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        print(f"\n🖼 پردازش: {filename}")

        results = reader.readtext(image_path, detail=1, paragraph=False)
        img_h = max([max([p[1] for p in box]) for _, box in results]) if results else 1

        texts = [(text, box) for (box, text, conf) in results]
        for t, _ in texts:
            print("-", t)

        extracted = categorize_texts(texts, img_h)

        print("\n📌 اطلاعات استخراج‌شده:")
        for k, v in extracted.items():
            print(f"{k}: {v}")

        with open(os.path.join(output_folder, f"{filename}.json"), 'w', encoding='utf-8') as f:
            json.dump(extracted, f, ensure_ascii=False, indent=2)
