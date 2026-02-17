from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os

from features import save_upload_file, get_fused_features
from model_loader import predict_class
from labels import CLASS_NAMES

app = FastAPI()

# ✅ ADD CORS HERE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # React dev server allowed
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Plant Disease API running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_path = save_upload_file(file)
    features = get_fused_features(image_path)
    class_id = predict_class(features)

    if os.path.exists(image_path):
        os.remove(image_path)

    return {
        "filename": file.filename,
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id]
    }
