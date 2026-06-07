from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import CSVData
import pandas as pd
import zipfile
import os
import matplotlib.pyplot as plt
from django.conf import settings
from .models import ImageDataset
from mainapp.utils.preprocess import preprocess_csv, preprocess_images
from mainapp.utils.train_ml_model import train_ml_model
from mainapp.utils.train_cnn_model import train_cnn_model
import tempfile
from django.contrib.auth.decorators import login_required, user_passes_test
from .models import SoilReport, SoilImage  # SoilReport model defined earlier
from .utils.report_parser import parse_csv_file, parse_pdf_file
from .utils.inference import predict_from_features
from .utils.cnn_inference import predict_soil_image
from .utils.inference import manual_predict_model
from django.core.files.storage import FileSystemStorage
from .utils.plant_cnn import predict_plant_disease
import cv2
import os, json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from django.conf import settings
from .utils.leaf_inference import predict_leaf
import uuid


MODEL_PATH = os.path.join(settings.MEDIA_ROOT, "ml_models/plant_disease_model.h5")
CLASS_PATH = os.path.join(settings.MEDIA_ROOT, "ml_models/class_indices.json")

model = None
class_map = None

def load_leaf_model():
    global model, class_map
    if model is None:
        model = load_model(MODEL_PATH)
        with open(CLASS_PATH, "r") as f:
            class_map = json.load(f)
    return model, class_map


@login_required
def leaf_predict(request):

    # 🔴 testimages folder OUTSIDE media
    test_folder = os.path.join(settings.BASE_DIR, "testimages")

    # Folder must exist
    if not os.path.exists(test_folder):
        return render(request, "mainapp/leaf_predict.html", {
            "error": "❌ testimages folder not found in project root."
        })

    if request.method == "POST" and "leaf" in request.FILES:
        uploaded_file = request.FILES["leaf"]
        filename = uploaded_file.name

        # 🔒 STRICT CHECK: file must already exist in testimages
        expected_path = os.path.join(test_folder, filename)

        if not os.path.exists(expected_path):
            return render(request, "mainapp/leaf_predict.html", {
                "error": "❌ Please select image ONLY from testimages folder"
            })

        # ✅ Use existing image directly
        disease, confidence = predict_leaf(expected_path)

        # ⚠️ Image URL (served via custom static handling)
        image_url = f"/testimages/{filename}"

        return render(request, "mainapp/leaf_result.html", {
            "image_url": image_url,
            "disease": disease,
            "confidence": confidence
        })

    return render(request, "mainapp/leaf_predict.html")

def landing_page(request):
    return render(request, "mainapp/landing.html")


def admin_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(username=username, password=password)

        if user is not None and (user.is_staff or user.is_superuser):
            login(request, user)
            return redirect("admin_dashboard")
        else:
            messages.error(request, "Invalid admin credentials")

    return render(request, "mainapp/admin_login.html")

@user_passes_test(lambda u: u.is_staff)
def admin_dashboard(request):
    return render(request, "mainapp/admin_dashboard.html")


def upload_csv(request):
    preview_html = None

    if request.method == "POST":
        csv_file = request.FILES.get("csv_file")

        if not csv_file.name.endswith(".csv"):
            return render(request, "mainapp/upload_csv.html",
                          {"error": "Please upload a valid CSV file."})

        # Save file
        data_obj = CSVData.objects.create(file=csv_file)

        # Read CSV file
        df = pd.read_csv(data_obj.file.path)

        # Convert first 5 rows to HTML table
        preview_html = df.head().to_html(classes="table table-bordered")

        return render(request, "mainapp/csv_preview.html",
                      {"preview": preview_html, "file_name": csv_file.name})

    return render(request, "mainapp/upload_csv.html")


def upload_image_dataset(request):
    graph_url = None
    class_counts = None

    if request.method == "POST":
        zip_file = request.FILES.get("zip_file")

        if not zip_file.name.endswith(".zip"):
            return render(request, "mainapp/upload_images.html",
                          {"error": "Please upload a valid ZIP file."})

        # Save ZIP
        obj = ImageDataset.objects.create(zip_file=zip_file)

        # Extract ZIP
        extract_path = os.path.join(settings.MEDIA_ROOT, "dataset/images/")
        os.makedirs(extract_path, exist_ok=True)

        with zipfile.ZipFile(obj.zip_file.path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

        obj.extracted_path = extract_path
        obj.save()

        # Count images per folder
        class_counts = {}
        for folder in os.listdir(extract_path):
            folder_path = os.path.join(extract_path, folder)
            if os.path.isdir(folder_path):
                count = len([f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
                class_counts[folder] = count

        # Generate bar graph
        plt.figure(figsize=(8, 5))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xlabel("Soil Class")
        plt.ylabel("Number of Images")
        plt.title("Image Dataset Distribution")
        plt.tight_layout()

        graph_path = os.path.join(settings.MEDIA_ROOT, "dataset/image_graph.png")
        plt.savefig(graph_path)
        plt.close()

        graph_url = settings.MEDIA_URL + "dataset/image_graph.png"

        return render(request, "mainapp/image_preview.html",
                      {"class_counts": class_counts, "graph": graph_url})

    return render(request, "mainapp/upload_images.html")


def preprocess_datasets(request):
    message_csv = ""
    message_img = ""

    if request.method == "POST":
        # CSV preprocessing
        status_csv, msg_csv = preprocess_csv()
        message_csv = msg_csv

        # Image preprocessing
        status_img, msg_img = preprocess_images()
        message_img = msg_img

        return render(request, "mainapp/preprocess_done.html",
                      {"msg_csv": message_csv, "msg_img": message_img})

    return render(request, "mainapp/preprocess.html")

def train_ml(request):
    train_acc = None
    test_acc = None
    message = ""

    if request.method == "POST":
        train_acc, test_acc, message = train_ml_model()
        return render(request, "mainapp/train_ml_done.html",
                      {"train": train_acc, "test": test_acc, "msg": message})

    return render(request, "mainapp/train_ml.html")


def train_cnn(request):
    train_acc = None
    val_acc = None
    message = ""

    if request.method == "POST":
        train_acc, val_acc, message = train_cnn_model()

        return render(request, "mainapp/train_cnn_done.html",
                      {"train": train_acc, "val": val_acc, "msg": message})

    return render(request, "mainapp/train_cnn.html")

def user_register(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password = request.POST.get("password")

        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already exists!")
            return redirect("user_register")

        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()
        messages.success(request, "Account created successfully! Please login.")
        return redirect("user_login")

    return render(request, "mainapp/register.html")


def user_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("user_dashboard")
        else:
            messages.error(request, "Invalid username or password")

    return render(request, "mainapp/login.html")


def user_logout(request):
    logout(request)
    return redirect("user_login")


def user_dashboard(request):
    if not request.user.is_authenticated:
        return redirect("user_login")

    return render(request, "mainapp/dashboard.html")


@login_required
def soil_manual_entry(request):
    if request.method == "POST":
        N = float(request.POST.get("N"))
        P = float(request.POST.get("P"))
        K = float(request.POST.get("K"))
        temp = float(request.POST.get("temperature"))
        humidity = float(request.POST.get("humidity"))
        ph = float(request.POST.get("ph"))
        rainfall = float(request.POST.get("rainfall"))

        # Create dictionary to pass to model
        values = {
            "N": N,
            "P": P,
            "K": K,
            "temperature": temp,
            "humidity": humidity,
            "ph": ph,
            "rainfall": rainfall,
        }

        # Model Prediction
        result = manual_predict_model(values)

        return render(request, "mainapp/manual_result.html", {
            "values": values,
            "prediction": result["label"],
            "confidence": result["confidence"]
        })

    return render(request, "mainapp/soil_manual.html")


@login_required
def soil_image_predict(request):

    # 🔒 STRICT FOLDER: testsoil (outside media)
    testsoil_folder = os.path.join(settings.BASE_DIR, "testsoil")

    # Folder must exist
    if not os.path.exists(testsoil_folder):
        messages.error(request, "❌ testsoil folder not found in project root.")
        return render(request, "mainapp/soil_image.html")

    if request.method == "POST":

        img = request.FILES.get("image")
        if not img:
            messages.error(request, "❌ Please select an image.")
            return redirect("soil_image_predict")

        filename = img.name
        expected_path = os.path.join(testsoil_folder, filename)

        # 🔴 STRICT CHECK: image must already exist in testsoil
        if not os.path.exists(expected_path):
            messages.error(
                request,
                "❌ Please select image ONLY from testsoil folder."
            )
            return redirect("soil_image_predict")

        # ✅ Predict directly from testsoil image
        soil_type, texture, fertility, recommendation, confidence = predict_soil_image(expected_path)

        image_url = f"/testsoil/{filename}"

        return render(request, "mainapp/soil_image_result.html", {
            "soil_type": soil_type,
            "texture": texture,
            "fertility": fertility,
            "recommendation": recommendation,
            "confidence": confidence,
            "image_path": image_url
        })

    return render(request, "mainapp/soil_image.html")


def plant_upload(request):
    if request.method == "POST":
        image = request.FILES["image"]
        fs = FileSystemStorage(location="media/plant_test")
        filename = fs.save(image.name, image)
        image_path = fs.path(filename)

        disease, confidence = predict_plant_disease(image_path)

        return render(request, "mainapp/plant_result.html", {
            "disease": disease,
            "confidence": confidence,
            "image_url": fs.url(filename)
        })

    return render(request, "mainapp/plant_upload.html")