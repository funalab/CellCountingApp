#!/bin/bash

python -W ignore tools/test_reg.py \
    --indir datasets/test \
    --outdir results/test \
    --init_model ccapp/app/core/models/reg.npz \
    --test_list datasets/split_list/test.list \
    --gpu -1 \
    --crop_size '(320, 320)' \
    --coordinate '(2100, 950)'