import os
import json
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = os.path.join(settings.MEDIA_ROOT, "ml_models/soil_cnn_model.h5")
CLASS_PATH = os.path.join(settings.MEDIA_ROOT, "ml_models/class_indices.json")

_model = None
_class_map = None

def load_cnn_model():
    global _model, _class_map
    
    # Load model
    if _model is None:
        _model = load_model(MODEL_PATH)

    # Load class mapping
    if _class_map is None:
        with open(CLASS_PATH, "r") as f:
            _class_map = json.load(f)  # {'Black_Soil': 0, 'Red_Soil': 1, ...}

    return _model, _class_map


def normalize_class_name(name):
    """Convert 'Black_Soil' → 'Black Soil' """
    return name.replace("_", " ").title()


def predict_soil_image(image_path):
    model, class_map = load_cnn_model()

    # Load image
    img = load_img(image_path, target_size=(150, 150))
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    preds = model.predict(img_arr)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])

    # Reverse mapping: index → class name
    inv_map = {v: k for k, v in class_map.items()}
    
    soil_raw = inv_map[class_idx]         # e.g. "Black_Soil"
    soil_type = normalize_class_name(soil_raw)  # e.g. "Black Soil"

    # Texture mapping
    texture_map = {
        "Black Soil": "Clayey & Moist",
        "Red Soil": "Fine-grained",
        "Clay": "Clayey",
        "Loam": "Loamy",
        "Sandy": "Sandy",
        "Alluvial": "Soft & Fertile",
    }
    texture = texture_map.get(soil_type, "Unknown")

    # Fertility mapping
    fertility_map = {
        "Black Soil": "High",
        "Alluvial": "High",
        "Loam": "Medium",
        "Clay": "Medium",
        "Red Soil": "Low",
        "Sandy": "Low",
    }
    fertility = fertility_map.get(soil_type, "Medium")

    # Recommended crops mapping
    recommendation_map = {
        "Black Soil": "Cotton, Soybean, Sorghum, Wheat, Maize",
        "Red Soil": "Millets, Groundnut, Pulses, Castor",
        "Alluvial": "Rice, Sugarcane, Wheat, Jute",
        "Clay": "Paddy, Wheat, Gram",
        "Loam": "Wheat, Maize, Pulses, Oilseeds",
        "Sandy": "Groundnut, Watermelon, Muskmelon",
    }
    recommendation = recommendation_map.get(soil_type, "General Crops")

    return soil_type, texture, fertility, recommendation, confidence
