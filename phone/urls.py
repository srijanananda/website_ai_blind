from django.urls import path
from . import views

urlpatterns = [
    path('', views.phone_home, name='phone_home'),
]
