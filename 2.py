yaml_content = """\
path: D:/card-id-detector
train: images/train
val: images/val

names:
  0: CARD
"""

with open("D:/card-id-detector/data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)

print("✅ فایل data.yaml ساخته شد.")
