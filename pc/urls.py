from django.urls import path
from .views import stream_page

urlpatterns = [
    path('', stream_page, name='pc_stream'),
]
