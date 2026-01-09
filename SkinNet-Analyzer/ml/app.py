import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

# =========================
# APP INIT
# =========================
app = FastAPI(title="SkinNet Analyzer API")

# =========================
# CORS (IMPORTANT)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "efficientnet.pth")

# =========================
# CLASS NAMES
# =========================
CLASS_NAMES = [
    "BA- cellulitis",
    "BA-impetigo",
    "FU-athlete-foot",
    "FU-nail-fungus",
    "FU-ringworm",
    "PA-cutaneous-larva-migrans",
    "VI-chickenpox",
    "VI-shingles"
]

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(CLASS_NAMES)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# =========================
# IMAGE TRANSFORMS
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
# STATUS API (Frontend uses this)
# =========================
@app.get("/api/status")
def status():
    return {"status": "ok"}

# =========================
# IMAGE UPLOAD & PREDICTION
# =========================
@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Upload JPG or PNG image only")

        image = Image.open(file.file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)

        return JSONResponse(content={
            "disease": CLASS_NAMES[predicted.item()],
            "severity": "Moderate",  # placeholder (frontend expects this)
            "confidence": round(confidence.item() * 100, 2)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
