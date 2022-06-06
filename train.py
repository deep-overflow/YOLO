import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import VOCDetection

import xmltodict
from PIL import Image
import numpy as np
from tqdm import tqdm


def train():
    # Config
    epochs = 10

    # Dataset
    train_dataset = YOLO_Dataset(
        root="data",
        year="2012",
        image_set="train",
        download=True
    )

    eval_dataset = YOLO_Dataset(
        root="data",
        year="2012",
        image_set="val",
        download=True
    )

    # DataLoader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=128,
                                  shuffle=True)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=128)

    for x, y in train_dataloader:
        print(x.shape)
        print(y.shape)
        break

    for x, y in eval_dataloader:
        print(x.shape)
        print(y.shape)
        break

    # Model
    net = YOLO()

    # Optimizer
    optimizer = optim.SGD(net.parameters(), lr=1e-2)

    for epoch in range(epochs):
        print(f"========== Epoch {epoch + 1} / {epochs} ==========")


train()
