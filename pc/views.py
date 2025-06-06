from django.shortcuts import render
from django.http import HttpResponse

def pc_home(request):
    return HttpResponse("Welcome to the PC page!")
