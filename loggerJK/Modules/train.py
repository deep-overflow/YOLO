import os
from pathlib import Path
from pprint import pprint
from collections import defaultdict

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

from datasets import *
from functions import *
from loss import *
from model import *

################### Argument Parsing ###################

parser = argparse.ArgumentParser(description=
                                 'YOLO V1'
                                 )    # 2. parser를 만든다.

# 3. parser.add_argument로 받아들일 인수를 추가해나간다.
# Poistional Arguments
parser.add_argument('S', help='S')    # 필요한 인수를 추가
parser.add_argument('B', help='B')
parser.add_argument('C', help='C')
# Optional Arguments
parser.add_argument('-s', '--seed', default = 42, help = 'seed value for reproducible output')
parser.add_argument('-h', '--height', default = 384, help = 'image height of model input')
parser.add_argument('-w', '--width', default = 384, help = 'image width of model input')
parser.add_argument('')




args = parser.parse_args()    # 4. 인수를 분석
# 출처: https://engineer-mole.tistory.com/213 [매일 꾸준히, 더 깊이:티스토리]

CONFIG = dict(
    S=7,
    B=2,
    C=20,
    seed=42,
    batch_size=4,
    epoch = 150,
    height=384,   # y
    width=384,    # x
    lambda_coord=5.,
    lambda_noobj=0.5,
    lr=1e-5,
    start_epoch=1,
    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    device=torch.device("mps" if torch.has_mps else "cpu"),
)
ic.disable()


################### WANDB ###################
import wandb
# If you don't want your script to sync to the cloud
os.environ['WANDB_MODE'] = 'offline'
os.environ['WANDB_API_KEY'] = 'd60a4af56f6cd9cccec7d9da1dbced7960b61310'

run = wandb.init(project="YOLOv1", entity="jiwon7258", config = CONFIG, job_type='train')
run.name = 'MPS'

################### SET SEED ###################
set_seed(CONFIG['seed'])


################### DATASET & DATALOADER ###################    
train_dataset_download = VOCDetection(root='./', year='2007',
                       image_set='train', download=True,)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=CONFIG['batch_size'], collate_fn=custom_collate_fn)
val_dataloader = DataLoader(dataset = val_dataset, batch_size=CONFIG['batch_size'], collate_fn=custom_collate_fn)

train_dataset = trainDataset()
val_dataset = valDataset()

# test_dataset_download = VOCDetection(root='./test/', year='2007', image_set='test', download=True)
# test_dataloader = DataLoader(dataset = test_dataset_download, batch_size=CONFIG['batch_size'], collate_fn=custom_collate_fn)




