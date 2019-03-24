from yahoo_weather import yahoo_weather as yw
from yahoo_weather.config.units import Unit
import numpy as np
from threading import Thread
import datetime, time
import json

units   = {'c': ['°C', 'km/h', 'mbar'], 'f': ['°F', 'm/h', 'inch Hg']}

codes = {0:'Tornado', 1:'Oluja', 2:'Uragan', 3:'Jaka oluja sa grmljavinom', 4:'Oluja sa grmljavinom',
         8:'Ledena kisa', 10:'Ledena kisa', 11:'Pljuskovi', 12:'Pljuskovi', 13:'Mecava', 15:'Mecava',
         17:'Grad', 20:'Magla', 36:'Vrucina', 43:'Jak sneg', 32:'Suncano'}

current_temp        = None
current_cond        = None
current_humid       = None
current_press       = None
current_vis         = None
current_wind        = None
current_code        = None
current_date        = None
current_datetime    = None
current_forecasts   = None

temp_diff   = 10
temp_max    = 35
temp_min    = -5

counter_t   = None
updater_t   = None
config      = None

running = False

warnings = {}


class MeteoWarning():

    def __init__(self, date, message):
        self.date = date
        self.message = message

    def __str__(self):
        return f'({str(self.date)}) - {self.message}'

    def __repr__(self):
        return f'({str(self.date)}) - {self.message}'

class Updater(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.name = 'Updater'
        self.running = True
        self.paused = True
        self.data = yw.YahooWeather(APP_ID=config['app_id'], apikey=config['client_id'], \
            apisecret=config['client_secret'])

    def run(self):
        while self.running:
            if self.paused:
                time.sleep(1)
            else:
                warnings.clear()
                self.get_weather()
                
                if current_code in codes.keys():
                    warnings.setdefault(current_date, []).append(MeteoWarning(current_date, \
                        f'Upozorenje! {codes[current_code]}.'))

                last_high = current_forecasts[0].high
                last_low = current_forecasts[0].low

                if last_high > temp_max:
                    warnings.setdefault(current_date, []).append(MeteoWarning(current_date, \
                        f'Visoka temperatura [{last_high}]. Ponesite flasu vode.'))
                    
                if last_low < temp_min:
                    warnings.setdefault(current_date, []).append(MeteoWarning(current_date, \
                        f'Visoka temperatura [{last_low}]. Pazite na led i ledenice.'))

                for i in range(1, len(current_forecasts)):
                    curr_high = current_forecasts[i].high
                    curr_low = current_forecasts[i].low
                    curr_date = str(current_forecasts[i].date.date())
                    if abs(last_high - curr_high) > temp_diff:
                        if last_high < curr_high:
                            warnings.setdefault(curr_date, []).append(MeteoWarning(curr_date, \
                                'Nagli porast temperature.'))
                        else: 
                            warnings.setdefault(curr_date, []).append(MeteoWarning(curr_date, \
                                'Nagli pad temperature.'))

                    if (abs(last_low - curr_low)) > temp_diff:
                        if last_low > curr_low:
                            warnings.setdefault(curr_date, []).append(MeteoWarning(curr_date, \
                                'Nagli pad temperature.'))
                        else:
                            warnings.setdefault(curr_date, []).append(MeteoWarning(curr_date, \
                                'Nagli porast temperature.'))
                    
                    if curr_high > temp_max:
                        warnings.setdefault(curr_date, []).append(MeteoWarning(curr_date, \
                            f'Visoka temperatura [{curr_high}]. Ponesite flasu vode sa sobom.'))

                    if curr_low < temp_min:
                        warnings.setdefault(curr_date, []).append(MeteoWarning(curr_date, \
                            f'Niska temperatura [{curr_low}]. Pazite na led i ledenice.'))

                    if current_forecasts[i].code in codes.keys():
                        warnings.setdefault(curr_date, []).append(MeteoWarning(curr_date, \
                            f'Upozorenje! {codes[current_forecasts[i].code]}.'))

                print(warnings)

                self.paused = True

    def get_weather(self):
        global current_temp, current_cond, current_humid, current_press, current_vis, \
            current_wind, current_code, current_date, current_datetime, current_forecasts

        self.data.get_yahoo_weather_by_city('belgrade', config['unit'])

        current_temp        = self.data.condition.temperature
        current_cond        = self.data.condition.text
        current_humid       = self.data.atmosphere.humidity
        current_press       = self.data.atmosphere.pressure
        current_vis         = self.data.atmosphere.visibility
        current_wind        = self.data.wind.speed
        current_code        = self.data.condition.code
        current_datetime    = str(datetime.datetime.today())
        current_date        = str(datetime.date.today())
        current_forecasts   = self.data.forecasts

    def stop(self):
        self.running = False

    def ping(self):
        self.paused = False    


class Counter(Thread):

    def __init__(self, timeout=1):
        Thread.__init__(self)
        self.name = 'Counter'
        self.running = True
        self.counter = 0
        self.timeout = timeout

    def run(self):
        while self.running:
            if self.counter == 0:
                updater_t.ping()

            print(self.counter)
            self.counter = (self.counter + 1) % self.timeout
            time.sleep(1)

    def stop(self):
        self.running = False

def get_temp():
    s = units[config['unit']][0]
    return f'{current_temp} {s}'

def meteo_init():
    global config, updater_t, counter_t, running

    with open('config.json', 'r') as f:
        config = json.load(f)

    updater_t = Updater()
    updater_t.start()

    counter_t = Counter(20)
    counter_t.start()

    running = True

def meteo_stop():
    global running

    running = False