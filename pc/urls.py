from django.urls import path
from . import views

urlpatterns = [
    path('', views.pc_home, name='pc_home'),
]
