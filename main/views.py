from django.shortcuts import render
from django.contrib import messages

# Create your views here.
def index(request):
    return render(request, 'index.html', {})

def mapa(request):
    import requests
    url = 'https://api.airvisual.com/v2/stations?city=Beijing&state=Beijing&country=China&key=wPHw9rvRRjYp9veR4'
    payload = {}
    headers = {}
    response = requests.request('GET', url, headers = headers, data = payload, allow_redirects=False, timeout=500)
    print(response.text)
    return render(request, 'mapa.html', {})