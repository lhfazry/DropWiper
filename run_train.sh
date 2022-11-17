#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python raindrop_detector.py \
    --data_dir=datasets/RainDrop \
    --batch_size=16 \
    --num_workers=4 \
    --accelerator=gpu \
    --variant=small \
    --max_epoch=50