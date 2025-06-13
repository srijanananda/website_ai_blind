#from django.urls import path
#from .views import stream_page, object_detect

#urlpatterns = [
#    path('', stream_page, name='pc_stream'),
 #   path('object_detect/', object_detect, name='object_detect'),
#]

#from django.urls import path
#from . import views
#
#urlpatterns = [
#    path('', views.index, name='index'),
#    path('start/', views.start_system, name='start_system'),
#    path('stop/', views.stop_system, name='stop_system'),
#    path('upload_face/', views.upload_face, name='upload_face'),
#    path('video_feed/', views.video_feed, name='video_feed'),
#    path('transcript/', views.get_transcript, name='get_transcript'),
#]

from django.urls import path
from . import views

urlpatterns = [
    path('', views.pc_dashboard),  # ← Fix here
    path('start_stream/', views.start_stream),
    path('stop_stream/', views.stop_stream),
    path('video_feed/', views.video_feed),
    path('set_mode/', views.set_mode),
    path('get_transcript/', views.get_transcript),
]
