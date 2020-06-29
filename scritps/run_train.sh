#!/bin/sh

python -W ignore tools/train_reg.py \
    --indir datasets/train \
    --outdir results/train \
    --train_list datasets/split_list/set1/train.list \
    --validation_list datasets/split_list/set1/validation.list \
    --gpu -1 \
    --epoch 100 \
    --batchsize 10 \
    --crop_size '(320, 320)' \
    --coordinate '(2100, 950)' \
    --optimizer Adam