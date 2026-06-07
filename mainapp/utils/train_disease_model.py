import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

DATASET_DIR = "C:/Users/sandeep/OneDrive/Desktop/soil_ai/dataset/PlantVillage"
MODEL_DIR = "mainapp/ml_models"
IMG_SIZE = 64
BATCH = 32

os.makedirs(MODEL_DIR, exist_ok=True)

train_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="training"
)

val_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation"
)

# CNN MODEL
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, epochs=10, validation_data=val_data)

# SAVE MODEL
model.save(os.path.join(MODEL_DIR, "leaf_disease_model.h5"))

# SAVE CLASS LABELS
with open(os.path.join(MODEL_DIR, "leaf_classes.json"), "w") as f:
    json.dump(train_data.class_indices, f)

print("✅ Leaf Disease Model Trained & Saved Successfully")
