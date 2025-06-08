import base64
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .ml import object_detection
from django.shortcuts import render

def stream_page(request):
    return render(request, 'pc/stream.html')


@csrf_exempt
def object_detect(request):
    if request.method == "POST":
        image_data_url = request.POST.get('image', '')
        if image_data_url.startswith('data:image/jpeg;base64,'):
            base64_str = image_data_url.split('data:image/jpeg;base64,')[1]
            image_bytes = base64.b64decode(base64_str)
            detected_object = object_detection.detect_objects_from_image_yolo(image_bytes)
            return JsonResponse({"object": detected_object})
        else:
            return JsonResponse({"object": ""})

    return JsonResponse({"error": "Invalid request method"}, status=400)
