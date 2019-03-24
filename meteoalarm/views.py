from django.shortcuts import render
from django.http import HttpResponse
from meteoalarm import meteo

class Warning():

    def __init__(self, date, message):
        self.date = date
        self.message = message

def meteo_view(request):
    if not meteo.running:
        meteo.meteo_init()

    
    
    return HttpResponse('TEST')