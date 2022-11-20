import argparse
import os
import os.path as osp
import torch
import cv2
import logging
import numpy as np
import shutil
from pathlib import Path
from glob import glob
from image_utils import center_crop, scale_image

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def resize_images(input_dir, output_dir):
    create_dir_if_not_exists(output_dir)

    files = glob(os.path.join(input_dir, "*.png"))

    for file in files:
        image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        resized = scale_image(center_crop(image, (480, 480)), 128/480)#center_crop(image, (128, 128))
        print(f"resized: {resized.shape}")
        cv2.imwrite(os.path.join(output_dir, Path(file).name), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare datasets')
    parser.add_argument("--input_dir", type=str, default="/workspace/DropWiper/datasets/RainDrop/val/data", help="Input directory")
    parser.add_argument("--output_dir", type=str, default="/workspace/DropWiper/datasets/RainDrop/val/resized", help="Output directory")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    resize_images(args.input_dir, args.output_dir)
