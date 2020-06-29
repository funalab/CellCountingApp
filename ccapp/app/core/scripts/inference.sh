#!/bin/sh

python -W ignore src/run.py \
    --input_a ../datasets/test/1.0_1/a.jpeg \
    --input_b ../datasets/test/1.0_1/b.jpeg \
    --init_model models/reg.npz \
    --gpu -1 \
    --crop_size '(640,640)' \
    --coordinate '(1840,700)'