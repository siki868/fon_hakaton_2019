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

def polen(request):
    import requests
    import json
    from yahoo_weather import yahoo_weather as yw
    from yahoo_weather.config.units import Unit
    app_id          = '8zRCxw38'
    client_id       = 'dj0yJmk9QXM1Z1JXNGdIN2ZnJnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PWEz'
    client_secret   = '9217a93b2c0e9245125cff2fcf98b1b327f82e3b'
    w =  [0.19066294, 0.11696213, -0.00772253]
    bias =  0.39566872
    data = yw.YahooWeather(APP_ID=app_id, apikey=client_id, apisecret=client_secret)
    locations = [[44.93019519, 20.51549239], [44.90669986, 20.39153628], [44.8070656, 20.39225383],
    [44.68198256, 20.50764152], [44.69895266, 20.40278736], [44.85624252, 20.45025023],
    [44.90019666, 20.37783463], [44.74309937, 20.34613769], [44.90692789, 20.45439552],
    [44.802134, 20.478144], [44.755132, 20.470130]]
    pollens = []
    for loc in locations:
        data.get_yahoo_weather_by_location(loc[0], loc[1], Unit.celsius)
        temp    = float(data.condition.temperature)
        hum  = float(data.atmosphere.humidity)/100
        wind   = float(data.wind.speed)

        pollen = temp*w[0] + wind*w[1] + hum*w[2] + bias
        loc.append(pollen)
    context = {
        "locations":locations,
        "w":w,
        "bias":bias,
    }
    return render(request, 'polen.html', context)

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

def zagadjenost(request):
    return render(request, 'zagadjenost.html')