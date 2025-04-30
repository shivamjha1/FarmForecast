# blog/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.homepage, name='homepage'),
    path('categories/', views.home, name='home'),
    #path('upload/', views.upload_data, name='upload-data'),
    #path('upload_file/visualizaations', views.file_visualize, name='file-visualize'),
    path('upload_file/<str:category>', views.file_upload_view, name='file-upload'),
    path('upload_csv/', views.upload_csv, name='upload_csv'),
    path('visualize/', views.visualize_csv, name='visualize_csv'),
    path('aboutus/',views.aboutus, name='aboutus')
]

