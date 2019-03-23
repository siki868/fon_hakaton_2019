from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout, authenticate

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
    return render(request, 'mapa.html')

def logout_request(request):
    logout(request)
    messages.info(request, f'Successfully logged out!')
    return redirect('main:index')

def login_request(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f'You are now logged in as {username}')
                return redirect('main:index')
            else:
                messages.info(request, 'Invalid credentials')
        else:
            messages.info(request, 'Invalid credentials')

    form = AuthenticationForm
    return render(request, 'login.html', {'form': form})

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'New account created: {username}')
            login(request, user)
            messages.info(request, f'You are now logged in as {username}')
            return redirect('main:index')
        else:
            for msg in form.error_messages:
                messages.error(request, f'{msg}: {form.error_messages[msg]}')
    form = UserCreationForm
    return render(request, 'register.html', {'form': form})