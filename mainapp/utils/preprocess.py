import os
import pandas as pd
import numpy as np
import shutil
import random
from sklearn.model_selection import train_test_split
from django.conf import settings

DATASET_CSV_PATH = os.path.join(settings.MEDIA_ROOT, "dataset/csv/")
IMAGE_DATASET_PATH = os.path.join(settings.MEDIA_ROOT, "dataset/images/")
PREPROCESS_DIR = os.path.join(settings.MEDIA_ROOT, "preprocessed/")

def preprocess_csv():
    # Load first CSV file in dataset folder
    csv_files = [f for f in os.listdir(DATASET_CSV_PATH) if f.endswith(".csv")]
    if not csv_files:
        return None, "No CSV dataset found!"

    csv_path = os.path.join(DATASET_CSV_PATH, csv_files[0])

    df = pd.read_csv(csv_path)

    # Basic cleaning: fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # 80/20 split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Create output directory
    os.makedirs(PREPROCESS_DIR, exist_ok=True)

    train_df.to_csv(os.path.join(PREPROCESS_DIR, "csv_train.csv"), index=False)
    test_df.to_csv(os.path.join(PREPROCESS_DIR, "csv_test.csv"), index=False)

    return True, "CSV preprocessing completed successfully."


def preprocess_images():
    if not os.path.exists(IMAGE_DATASET_PATH):
        return None, "Image dataset folder not found!"

    train_dir = os.path.join(PREPROCESS_DIR, "images/train/")
    test_dir = os.path.join(PREPROCESS_DIR, "images/test/")

    # Clear old preprocessed data
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Loop through each image class
    for class_name in os.listdir(IMAGE_DATASET_PATH):
        class_folder = os.path.join(IMAGE_DATASET_PATH, class_name)

        if os.path.isdir(class_folder):
            images = [img for img in os.listdir(class_folder) if img.lower().endswith((".jpg", ".png", ".jpeg"))]

            # 80/20 split
            random.shuffle(images)
            split_index = int(len(images) * 0.8)
            train_imgs = images[:split_index]
            test_imgs = images[split_index:]

            # Create class folders
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # Copy files
            for img in train_imgs:
                shutil.copy(os.path.join(class_folder, img), os.path.join(train_dir, class_name, img))

            for img in test_imgs:
                shutil.copy(os.path.join(class_folder, img), os.path.join(test_dir, class_name, img))

    return True, "Image preprocessing completed successfully."
