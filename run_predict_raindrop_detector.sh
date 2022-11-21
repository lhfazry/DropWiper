#!/bin/bash
CUDA_VISIBLE_DEVICES=$1 python raindrop_detector.py \
    --mode=predict \
    --data_dir=datasets/RainDrop \
    --ckpt_path=lightning_logs/raindrop_detector/version_13/checkpoints/epoch\=45-step\=1242.ckpt \
    --batch_size=32 \
    --num_workers=4 \
    --accelerator=gpu