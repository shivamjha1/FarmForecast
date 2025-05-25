from django.contrib import admin

# Register your models here.
from .models import Crop, CSVFile, Fruit  # Replace with your actual model

admin.site.register(Crop)
admin.site.register(CSVFile)
admin.site.register(Fruit)