from django.urls import path
from . import views

app_name = 'app'

urlpatterns = [
    path('', views.uploader, name='uploader'),
    path('uploader', views.uploader,name='uploader'),
]
