import os
import cv2
import numpy as np
from django.conf import settings
from tensorflow.keras.models import model_from_json

MODEL_DIR = os.path.join(settings.BASE_DIR, "plant_model")

PLANTS = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus'
]

_loaded_model = None

def load_model_once():
    global _loaded_model
    if _loaded_model is None:
        with open(os.path.join(MODEL_DIR, "model.json"), "r") as f:
            _loaded_model = model_from_json(f.read())
        _loaded_model.load_weights(os.path.join(MODEL_DIR, "model_weights.h5"))
    return _loaded_model


def predict_plant_disease(image_path):
    model = load_model_once()

    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (1, 64, 64, 3))

    preds = model.predict(img)
    idx = np.argmax(preds)
    confidence = float(preds[0][idx])

    return PLANTS[idx], confidence
