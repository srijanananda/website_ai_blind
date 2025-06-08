from django.shortcuts import render
from django.http import HttpResponse

def stream_page(request):
    return render(request, 'pc/stream.html')
