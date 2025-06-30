from pathlib import Path

images_train = list(Path("images/train").glob("*.jpg"))
labels_all = list(Path("annotations/labels_all").glob("*.txt"))

matched = [img for img in images_train if (Path("annotations/labels_all") / (img.stem + ".txt")).exists()]
print(f"از {len(images_train)} عکس در train فقط {len(matched)} تا فایل برچسب دارن.")
