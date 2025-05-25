from django.db import models

# Create your models here.

class Crop(models.Model):
    model_name=models.CharField(max_length=30)
    state_encoded=models.CharField(max_length=30,null=True)
    commodity_encoded=models.CharField(max_length=30,null=True)
    

class CSVFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

class Fruit(models.Model):
    state = models.CharField(max_length=100)
    commodity = models.CharField(max_length=100)
    arrival_date = models.DateField()
    min_price = models.DecimalField(max_digits=10, decimal_places=2)
    max_price = models.DecimalField(max_digits=10, decimal_places=2)
    modal_price = models.DecimalField(max_digits=10, decimal_places=2)
    commodity_code = models.CharField(max_length=50)