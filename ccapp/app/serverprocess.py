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
from .core.src.run import *


if __name__ == '__main__':
    ARG_REQUIREMENT_N = 2
    args = sys.argv

    if ARG_REQUIREMENT_N == len(args):
        if os.path.exists(args[0]) is True and os.path.exists(args[1]) is True:
            _starttime = time.time()
            result = run(settings.BASE_DIR + download_url_list[0], settings.BASE_DIR + download_url_list[1], -1)
            _elapstime = time.time() - _starttime
        else:
            print('err @ os.path.exists')
    else:
        print('err @ ARG_REQUIREMENT_N')


context = dict()
if request.method == 'POST':
    form = UploadFormSet(request.POST,request.FILES)
    if form.is_valid():
        download_url_list = form.save()
        result = run(settings.BASE_DIR + download_url_list[0], settings.BASE_DIR + download_url_list[1], -1)
        context = {'download_url_list': download_url_list,
                   'form': UploadFormSet(),
                   'result': result,
                   }
        return render(request, 'app/result.html', context)
    else:
        download_url_list = 'NONE'
        context = {'form': form}
        return render(request, 'app/uploader.html', context)
else:
    form = UploadFormSet()
    context = {'form': form}
