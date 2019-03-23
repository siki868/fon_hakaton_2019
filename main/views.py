from django.shortcuts import render
from django.contrib import messages

# Create your views here.
def index(request):
    import requests
    import json
    url = 'https://api.airvisual.com/v2/nearest_city?key=wPHw9rvRRjYp9veR4'
    payload = {}
    headers = {}
    response = requests.request('GET', url, headers = headers, data = payload, allow_redirects=False, timeout=500)
    j = json.loads(response.text)
    location = j["data"]["city"]
    temperature = j["data"]["current"]["weather"]["tp"]
    pressure = j["data"]["current"]["weather"]["pr"]
    humidity = j["data"]["current"]["weather"]["hu"]
    wind_speed = j["data"]["current"]["weather"]["ws"]
    wind_direction = j["data"]["current"]["weather"]["wd"]
    icon = j["data"]["current"]["weather"]["ic"]
    print(icon)
    aqius = j["data"]["current"]["pollution"]["aqius"]
    context = {"location":location,
        "temperature":temperature,
        "pressure":pressure,
        "humidity":humidity,
        "wind_speed":wind_speed,
        "wind_direction":wind_direction, 
        "aqius":aqius,
        "icon":icon}

    return render(request, 'index.html', context)

def mapa(request):
    return render(request, 'mapa.html', context)