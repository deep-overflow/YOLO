"""  
TODO
- 
"""
import os
from pathlib import Path
from pprint import pprint
from collections import defaultdict
from xml.etree.ElementTree import Comment

# XML Parsing
import xmltodict
import json

# Pytorch
import torch
import torch.nn as nn
import torchvision.datasets as transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset, DataLoader
import timm
from einops import rearrange, reduce, repeat
from torchsummary import summary


# Image Processing
import cv2
import numpy as np
from PIL import Image, ImageDraw
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


# Debugging
from icecream import ic

# math, plotting
import matplotlib.pyplot as plt
from math import sqrt
import random

from tqdm import tqdm

# Arguments
import argparse

from config import get_config
from functions import bbox_transform, set_seed, IOU, print_img
from datasets import build_dataset
from model import build_model
from trainer import run_training


if __name__ == '__main__':
    # -----------------------------------------------------------------
    # Argument Parsing
    # ------------------------------------------------------------------

    parser = argparse.ArgumentParser(description='YOLO V1 by Jiwon Kang (loggerJK)'
                                     )

    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file', )
    args = parser.parse_args()
    # 출처: https://engineer-mole.tistory.com/213 [매일 꾸준히, 더 깊이:티스토리]

    # ------------------------------------------------------------------
    # CONFIG
    # ----------------------------------------------------------------
    config, config_dict = get_config(args=args)

    # ----------------------------------------------------------------
    # wandb
    # ------------------------------------------------------------------
    if (config.WANDB.USE):
        import wandb
        # If you don't want your script to sync to the cloud
        # os.environ['WANDB_MODE'] = 'offline'

        run = wandb.init(project="YOLOv1", entity="jiwon7258",
                         config=config_dict, job_type='train')

   # ------------------------------------------------------------------
   # SET SEED
   # ------------------------------------------------------------------
    set_seed(config.MISC.SEED)

   # ------------------------------------------------------------------
   # DATASET & DATALOADER
   # ------------------------------------------------------------------
    train_dataset, val_dataset, test_dataset, trainval_dataset, train_dataloader, val_dataloader, test_dataloader, trainval_dataloader = build_dataset(
        config)

   # --------------------------------------------------------
   # BUILD MODEL
   # --------------------------------------------------------
    model = build_model(config)
    model.to(config.TRAINING.DEVICE)

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=config.TRAINING.LR)

   # --------------------------------------------------------
   # TODO
   # SCHEDULER
   # --------------------------------------------------------

    # --------------------------------------------------------
    # Run Training
    # --------------------------------------------------------
    print('### Using Device {0} ###'.format(config.TRAINING.DEVICE))
    run_training(config=config, model=model, train_dataloader=train_dataloader,
                 val_dataloader=val_dataloader, optimizer=optimizer, scheduler=None)
