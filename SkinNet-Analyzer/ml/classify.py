import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientnet.pth")

# =========================
# CLASS NAMES (MUST MATCH TRAINING)
# =========================
CLASS_NAMES = [
    'BA- cellulitis',
    'BA-impetigo',
    'FU-athlete-foot',
    'FU-nail-fungus',
    'FU-ringworm',
    'PA-cutaneous-larva-migrans',
    'VI-chickenpox',
    'VI-shingles'
]

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
def load_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        len(CLASS_NAMES)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# =========================
# IMAGE TRANSFORM
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
# PREDICTION FUNCTION
# =========================
def predict(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError("Image not found")

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return {
        "prediction": CLASS_NAMES[predicted.item()],
        "confidence": round(confidence.item() * 100, 2)
    }

# =========================
# TEST LOCALLY
# =========================
if __name__ == "__main__":
    result = predict("test.jpg")
    print(result)
