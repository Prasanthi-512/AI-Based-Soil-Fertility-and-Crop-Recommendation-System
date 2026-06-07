import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from django.conf import settings

PREPROCESS_DIR = os.path.join(settings.MEDIA_ROOT, "preprocessed/images/")
MODEL_DIR = os.path.join(settings.MEDIA_ROOT, "ml_models/")

def train_cnn_model():

    train_dir = os.path.join(PREPROCESS_DIR, "train")
    test_dir = os.path.join(PREPROCESS_DIR, "test")

    # Validate dataset exists
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        return None, None, "Error: Train folder missing or empty!"

    if not os.path.exists(test_dir) or not os.listdir(test_dir):
        return None, None, "Error: Test folder missing or empty!"

    # Generators
    train_gen = ImageDataGenerator(rescale=1/255.0)
    test_gen = ImageDataGenerator(rescale=1/255.0)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(150,150),
        batch_size=32,
        class_mode="categorical"
    )

    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=(150,150),
        batch_size=32,
        class_mode="categorical"
    )

    # CLASS MAPPING
    class_indices = train_data.class_indices  # { 'Black_Soil':0, 'Red_Soil':1 ... }

    # CNN MODEL
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(len(class_indices), activation='softmax')
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # TRAIN
    history = model.fit(
        train_data,
        epochs=10,
        validation_data=test_data
    )

    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]

    # SAVE MODEL
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "soil_cnn_model.h5"))

    # SAVE CLASS INDEX JSON
    CLASS_PATH = os.path.join(MODEL_DIR, "class_indices.json")
    with open(CLASS_PATH, "w") as f:
        json.dump(class_indices, f)

    return train_acc, val_acc, "CNN Model Training Completed Successfully!"
