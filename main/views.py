from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages

# Create your views here.
def index(request):
    return render(request, 'index.html', {})

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