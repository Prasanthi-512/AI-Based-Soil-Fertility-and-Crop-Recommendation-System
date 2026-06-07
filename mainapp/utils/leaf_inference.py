import os, json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = "mainapp/ml_models/leaf_disease_model.h5"
CLASS_PATH = "mainapp/ml_models/leaf_classes.json"
IMG_SIZE = 64

model = load_model(MODEL_PATH)

with open(CLASS_PATH) as f:
    class_map = json.load(f)

labels = {v:k for k,v in class_map.items()}

def predict_leaf(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)[0]
    idx = np.argmax(preds)

    return labels[idx], float(preds[idx])
