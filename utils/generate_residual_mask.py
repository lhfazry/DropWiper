import argparse
import os
import os.path as osp
import cv2
import logging
import numpy as np
from pathlib import Path
from glob import glob

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def generate_residual_mask(input_dir):
    output_dir = os.path.join(input_dir, 'mask')
    create_dir_if_not_exists(output_dir)

    data = glob(os.path.join(input_dir, 'data', '*.png'))
    gts = glob(os.path.join(input_dir, 'gt', '*.png'))

    for idx in range(len(data)):
        print(f"Processing: {data[idx]}, {gts[idx]}")
        rain_img = cv2.cvtColor(cv2.imread(data[idx]), cv2.COLOR_BGR2GRAY)
        clean_img = cv2.cvtColor(cv2.imread(gts[idx]), cv2.COLOR_BGR2GRAY)

        res = rain_img - clean_img
        res = np.where(res > 30, 255, 0)

        cv2.imwrite(os.path.join(output_dir, Path(data[idx]).name), res)
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate residual mask')
    parser.add_argument("--input_dir", type=str, default="/workspace/DropWiper/datasets/RainDrop/train", help="Input directory")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    generate_residual_mask(args.input_dir)
