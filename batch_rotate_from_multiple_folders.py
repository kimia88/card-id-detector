import cv2
import pytesseract
import os
from pathlib import Path
from tqdm import tqdm

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

INPUT_DIRS = [
    r"D:\card-id-detector\runs\detect\predict3",
    r"D:\card-id-detector\runs\detect\predict4"
]
OUTPUT_DIR = r"D:\card-id-detector\rotated_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def rotate_image(image, angle):
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image

def is_face_top_left(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    h, w = gray.shape
    for (x, y, fw, fh) in faces:
        cx, cy = x + fw//2, y + fh//2
        if cx < w//2 and cy < h//2:
            return True
    return False

def count_farsi_chars(text):
    return sum(1 for ch in text if '\u0600' <= ch <= '\u06FF')

def get_farsi_score(image):
    h = image.shape[0]
    top_half = image[0:h//2, :]
    bottom_half = image[h//2:, :]

    text_top = pytesseract.image_to_string(top_half, lang='fas', config='--psm 6')
    text_bottom = pytesseract.image_to_string(bottom_half, lang='fas', config='--psm 6')

    score = count_farsi_chars(text_top) * 1.5 + count_farsi_chars(text_bottom)
    return score

for folder in INPUT_DIRS:
    image_files = list(Path(folder).rglob("*.[jp][pn]g")) + list(Path(folder).rglob("*.jpeg"))
    for image_path in tqdm(image_files, desc=f"Processing {folder}"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        best_score = -1
        best_angle = 0
        best_image = image

        for angle in [0, 90, 180, 270]:
            rotated = rotate_image(image, angle)
            has_face_top_left = is_face_top_left(rotated)
            ocr_score = get_farsi_score(rotated)
            total_score = ocr_score + (10 if has_face_top_left else 0)

            if total_score > best_score:
                best_score = total_score
                best_angle = angle
                best_image = rotated

        print(f"{image_path.name}: زاویه انتخاب شده = {best_angle}")
        output_name = f"{image_path.stem}_rotated.jpg"
        cv2.imwrite(str(Path(OUTPUT_DIR) / output_name), best_image)
