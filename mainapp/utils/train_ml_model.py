import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from django.conf import settings

MODEL_DIR = os.path.join(settings.MEDIA_ROOT, "ml_models/")
CSV_TRAIN_PATH = os.path.join(settings.MEDIA_ROOT, "preprocessed/csv_train.csv")
CSV_TEST_PATH = os.path.join(settings.MEDIA_ROOT, "preprocessed/csv_test.csv")

def train_ml_model():
    # Check CSV files
    if not os.path.exists(CSV_TRAIN_PATH):
        return None, None, "CSV training file missing!"
    if not os.path.exists(CSV_TEST_PATH):
        return None, None, "CSV testing file missing!"

    # Load datasets
    train_df = pd.read_csv(CSV_TRAIN_PATH)
    test_df = pd.read_csv(CSV_TEST_PATH)

    # Assume last column is Label (Low/Medium/High or crop)
    label_column = train_df.columns[-1]

    X_train = train_df.drop(label_column, axis=1)
    y_train = train_df[label_column]

    X_test = test_df.drop(label_column, axis=1)
    y_test = test_df[label_column]

    # ML Pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)

    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(MODEL_DIR, "fertility_model.pkl")
    joblib.dump(pipeline, model_path)

    return train_acc, test_acc, "Model training completed successfully!"
