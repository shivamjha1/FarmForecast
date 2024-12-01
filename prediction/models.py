from django.db import models

# Create your models here.

class Crop(models.Model):
    model_name=models.CharField(max_length=30)
    state_encoded=models.CharField(max_length=30,null=True)
    commodity_encoded=models.CharField(max_length=30,null=True)
    

class CSVFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
