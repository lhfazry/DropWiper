#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python raindrop_detector.py \
    --data_dir=datasets/RainDropS_valid/ \
    --batch_size=32 \
    --num_workers=4 \
    --accelerator=gpu \
    --max_epoch=100