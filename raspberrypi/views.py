from django.shortcuts import render
from django.http import HttpResponse

def pi_home(request):
    return HttpResponse("Welcome to the Raspberry Pi page!")
