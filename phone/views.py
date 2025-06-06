from django.shortcuts import render
from django.http import HttpResponse

def phone_home(request):
    return HttpResponse("Welcome to the Phone page!")
