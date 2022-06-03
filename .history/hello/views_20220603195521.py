from django.shortcuts import redirect, render
from django.http import HttpResponse
import json

from .models import Greeting
from django.views.decorators.csrf import csrf_protect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import User, Data_ThuatToan, Upload_File
from hello.static.algorithm import MODEL 

import os
# Create your views here.se


def index(request):
    if('username' in request.session):
        return render(request, 'home.html')
    else:
        return redirect('login')


def test(request):

    return render(request, "testpage.html")


def svm_imoc(request):

    return render(request, "svm_imoc.html")


def login(request):
    if request.method == 'POST':
        username = request.POST["Username"]
        password = request.POST["Password"]
        admin = {
            'username': username,
            'password': password
        }
        request.session['username'] = username
        request.session['password'] = password
        return redirect('home')
    else:
        return render(request, 'login.html', {'users': User.objects.all()})


def logout(request):
    request.session.flush()
    return redirect('/')


def auth_login(request):
    # if(request.method == 'POST'):
    #     print(request.POST)
    # else:
    return render(request, "login.html")


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'uploadFile/simple_upload.html', {
            'uploaded_file_url': uploaded_file_url
        })
    return render(request, 'pages/simple_upload.html')


def process_test(request):
    if request.method == 'POST':
        myfile = request.FILES['file-csv-open']
        fs = FileSystemStorage(settings.MEDIA_ROOT+'/media/')
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        # print(uploaded_file_url)
        array_nu, MODEL(uploaded_file_url, settings.STATIC_ROOT,10,3)
        return render(request, 'home.html', {
            'uploaded_file_url': uploaded_file_url
        })
    else:
        return redirect('home')
