from django.urls import path
from . import views

urlpatterns = [
    path("", views.landing_page, name="landing_page"),
    path("admin_login/", views.admin_login, name="admin_login"),
    path("admin_dashboard/", views.admin_dashboard, name="admin_dashboard"),
    path("admin_upload_csv/", views.upload_csv, name="upload_csv"),
    path("admin_upload_images/", views.upload_image_dataset, name="upload_images"),
    path("admin_preprocess/", views.preprocess_datasets, name="preprocess"),
    path("admin_train_ml/", views.train_ml, name="train_ml"),
    path("admin_train_cnn/", views.train_cnn, name="train_cnn"),
    path("register/", views.user_register, name="user_register"),
    path("login/", views.user_login, name="user_login"),
    path("logout/", views.user_logout, name="user_logout"),
    path("dashboard/", views.user_dashboard, name="user_dashboard"),
    path("soil_manual/", views.soil_manual_entry, name="soil_manual"),
    path("soil_image/", views.soil_image_predict, name="soil_image_predict"),
    path("plant_disease/", views.plant_upload, name="plant_disease"),
    path("leaf_predict/", views.leaf_predict, name="leaf_predict"),
    
]
