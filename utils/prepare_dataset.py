import argparse
import os
import os.path as osp
import pandas as pd
import torch
import cv2
import logging
import imutils
import numpy as np
import shutil
from pathlib import Path
from glob import glob
from sklearn.model_selection import train_test_split

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def prepare_dataset(input_dir, output_dir):
    split = ['train', 'val']
    subdir = ['data', 'masks']

    create_dir_if_not_exists(output_dir)

    for sp in split:
        create_dir_if_not_exists(os.path.join(output_dir, sp))

        for sd in subdir:
            create_dir_if_not_exists(os.path.join(output_dir, sp, sd))

    data = glob(os.path.join(input_dir, "*_B.png")).sort()
    masks = glob(os.path.join(input_dir, "*_M.png")).sort()

    data_train, data_val, masks_train, masks_val = train_test_split(data, masks, 
        test_size=0.2, random_state=42)

    for item_data, item_mask,  in zip(data_train, masks_train):
        filename = Path(item_data).name.replace('_B', '')
        shutil.copy(item_data, os.path.join('train', 'data', filename))
        shutil.copy(item_mask, os.path.join('train', 'masks', filename))

    for item_data, item_mask,  in zip(data_val, masks_val):
        filename = Path(item_data).name.replace('_B', '')
        shutil.copy(item_data, os.path.join('val', 'data', filename))
        shutil.copy(item_mask, os.path.join('val', 'masks', filename))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare datasets')
    parser.add_argument("--input_dir", type=str, default="/workspace/DropWiper/datasets/RainDropS/", help="Input directory")
    parser.add_argument("--output_dir", type=str, default="/workspace/DropWiper/datasets/RainDrop/", help="Output directory")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    prepare_dataset(args.input_dir, args.output_dir)
