#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python raindrop_detector.py \
    --mode=predict \
    --data_dir=datasets/RainDrop/test_a \
    --batch_size=32 \
    --num_workers=4 \
    --accelerator=gpu