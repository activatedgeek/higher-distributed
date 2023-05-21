#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 \
accelerate launch --multi_gpu train_toy.py --log-dir=.log/gpus-1

CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch --multi_gpu train_toy.py --log-dir=.log/gpus-2

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --multi_gpu train_toy.py --log-dir=.log/gpus-4

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --multi_gpu train_toy.py --log-dir=.log/gpus-8
