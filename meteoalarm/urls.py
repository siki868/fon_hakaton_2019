from . import views
from django.urls import path

app_name = 'meteoalarm'

urlpatterns = [
    path('meteoalarm/', views.meteo_view, name='meteoalarm')
]