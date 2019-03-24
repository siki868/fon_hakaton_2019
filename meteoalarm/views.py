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
    upozorenja = meteo.warnings
    upoz = []
    for key in upozorenja:
        for k in upozorenja[key]:
            upoz.append([k.date, k.level, k.message[11:-1]])
    print(upoz)
    return render(request, 'meteo_main.html', {"upozorenja":upoz})