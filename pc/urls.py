from django.urls import path
from .views import stream_page, object_detect

urlpatterns = [
    path('', stream_page, name='pc_stream'),
    path('object_detect/', object_detect, name='object_detect'),
]
