from django.urls import path
from . import views

urlpatterns = [
    path('', views.pi_home, name='pi_home'),
]
