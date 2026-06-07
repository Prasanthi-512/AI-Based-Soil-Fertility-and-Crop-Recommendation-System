from django.db import models
from django.contrib.auth.models import User

class CSVData(models.Model):
    file = models.FileField(upload_to='dataset/csv/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.file)


class ImageDataset(models.Model):
    zip_file = models.FileField(upload_to='dataset/images_zip/')
    extracted_path = models.CharField(max_length=255, null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)


class SoilReport(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='user_reports/')
    ph = models.FloatField(null=True, blank=True)
    n = models.FloatField(null=True, blank=True)
    p = models.FloatField(null=True, blank=True)
    k = models.FloatField(null=True, blank=True)
    oc = models.FloatField(null=True, blank=True)
    ec = models.FloatField(null=True, blank=True)
    moisture = models.FloatField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

class SoilImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='user_images/')
    soil_type = models.CharField(max_length=50, null=True)
    texture = models.CharField(max_length=50, null=True)
    fertility = models.CharField(max_length=20, null=True)
    recommendation = models.TextField(null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
