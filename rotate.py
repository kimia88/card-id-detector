import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim

# مسیر دیتاست (مسیر دقیق خودت رو قرار بده)
DATA_DIR = r"D:\card-id-detector\dataset"

# تبدیل‌ها (augmentations ساده)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# دیتاست و داتالودر
dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# دستگاه (GPU اگر باشه)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# مدل pretrained ResNet18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 کلاس (0، 90، 180، 270)
model = model.to(device)

# Loss و Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# آموزش مدل
num_epochs = 20
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataset)
    epoch_acc = running_corrects.double() / len(dataset)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# ذخیره مدل
torch.save(model.state_dict(), "rotation_classifier.pth")
print("Training complete and model saved!")
