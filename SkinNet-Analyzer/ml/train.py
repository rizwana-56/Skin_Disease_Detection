import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

# =========================
# BASE PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ FIXED DATASET PATH (MATCHES YOUR FOLDERS)
DATASET_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "skin-disease-datasaet",
    "train_set"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# CHECK DATASET PATH
# =========================
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"❌ Dataset path not found: {DATASET_DIR}")

print("✅ Dataset found at:", DATASET_DIR)

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# DATASET & LOADER
# =========================
train_dataset = datasets.ImageFolder(
    root=DATASET_DIR,
    transform=transform
)

num_classes = len(train_dataset.classes)

print("Detected classes:", train_dataset.classes)
print("Number of classes:", num_classes)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0  # Windows safe
)

# =========================
# TRAIN FUNCTION
# =========================
def train_model(model, save_path, is_regression=False, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            if is_regression:
                labels = labels.float().unsqueeze(1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved: {save_path}\n")

# =========================
# EfficientNet (Classifier)
# =========================
efficientnet = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)
efficientnet.classifier[1] = nn.Linear(
    efficientnet.classifier[1].in_features,
    num_classes
)

train_model(
    efficientnet,
    os.path.join(MODEL_DIR, "efficientnet.pth")
)

# =========================
# ResNet (Classifier)
# =========================
resnet = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)
resnet.fc = nn.Linear(
    resnet.fc.in_features,
    num_classes
)

train_model(
    resnet,
    os.path.join(MODEL_DIR, "resnet.pth")
)

# =========================
# MobileNet (Classifier)
# =========================
mobilenet = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.DEFAULT
)
mobilenet.classifier[3] = nn.Linear(
    mobilenet.classifier[3].in_features,
    num_classes
)

train_model(
    mobilenet,
    os.path.join(MODEL_DIR, "mobilenet.pth")
)

# =========================
# Severity Model (Regression)
# =========================
severity_model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)
severity_model.classifier[1] = nn.Linear(
    severity_model.classifier[1].in_features,
    1
)

train_model(
    severity_model,
    os.path.join(MODEL_DIR, "severity_model.pth"),
    is_regression=True
)

print("🎉 ALL MODELS TRAINED SUCCESSFULLY")
