from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.views import generic
from .forms import *
from .models import *

# add
from django.conf import settings
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

import sys
import time

from .core.src.run import *

# /add ここまでが追加分
def home(request):
    return render(request, 'app/home.html')

def uploader(request):
    _start_time = time.time()
    context = dict()
    if request.method == 'POST':
        form = UploadFormSet(request.POST, request.FILES)
        if form.is_valid():
            download_url_list = form.save()
            _download_time = time.time()
            _elapse = _download_time - _start_time
            print('Images DL Time', _elapse, '(s)')
            print('url list a: {}'.format(download_url_list[0]))
            print('url list b: {}'.format(download_url_list[1]))
            result = run(settings.BASE_DIR + download_url_list[1], settings.BASE_DIR + download_url_list[0], -1)
            _modelrun_time = time.time()
            _elapse = _modelrun_time - _download_time
            print('Counting Time by Model(RunningTime:core.src.run)', _elapse, '(s)')
            # result = tokuoka_function(settings.BASE_DIR + download_url_list[0], settings.BASE_DIR + download_url_list[1])
            context = {'result': np.round(result, 3),
                       # 'download_url_list': download_url_list,
                       # 'form': Uplo adFormSet(),
                      }
            _createcontext_time = time.time()
            _elapse = _createcontext_time - _modelrun_time
            print('Rendering Time', _elapse, '(s)')
            return render(request, 'app/result.html', context)
        else:
            download_url_list = 'NONE'
            context = {'form': form}
            return render(request, 'app/uploader.html', context)
    else:
        form = UploadFormSet()
        context = {'form': form}
    return render(request, 'app/uploader.html', context)
