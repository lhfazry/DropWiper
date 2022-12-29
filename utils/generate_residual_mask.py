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

def generate_residual_mask(input_dir, threshold):
    splits = ['train', 'val', 'test']

    for split in splits:
        output_dir = os.path.join(input_dir, split, 'mask')
        create_dir_if_not_exists(output_dir)

        data = glob(os.path.join(input_dir, split, 'data', '*.png'))
        gts = glob(os.path.join(input_dir, split, 'gt', '*.png'))

        data.sort()
        gts.sort()

        for idx in range(len(data)):
            #print(f"Processing: {data[idx]}, {gts[idx]}")
            rain_img = cv2.cvtColor(cv2.imread(data[idx]), cv2.COLOR_BGR2GRAY)
            clean_img = cv2.cvtColor(cv2.imread(gts[idx]), cv2.COLOR_BGR2GRAY)

            res = rain_img - clean_img
            res = np.where(res > threshold, 0, 255)

            cv2.imwrite(os.path.join(output_dir, Path(data[idx]).name), res)
    
def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate residual mask')
    parser.add_argument("--input_dir", type=str, default="/workspace/DropWiper/datasets/RainDrop_resized/", help="Input directory")
    parser.add_argument("--threshold", type=int, default=200, help="threshold")
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    split = ['train', 'test', 'val']

    for item_split in split:
        generate_residual_mask(os.path.join(args.input_dir, item_split), args.threshold)
