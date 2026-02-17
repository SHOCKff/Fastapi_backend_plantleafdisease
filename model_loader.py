import os
import numpy as np
from joblib import load

MODEL_PATH = "model.joblib"

_model = None


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("model.joblib not found")
        _model = load(MODEL_PATH)
        print("✅ SVM model loaded")
        print("MODEL CLASSES:", _model.classes_)   # <-- add this
    return _model

def predict_class(features: np.ndarray):
    model = load_model()
    features = features.reshape(1, -1)
    pred = model.predict(features)[0]
    return int(pred)
