import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2)

        self.activation = nn.LeakyReLU()

    def forward(self, x):
        return self.activation(self.conv(x))


class Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.module = nn.Sequential(
            Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1),
            Conv(
                in_channels=out_channels,
                out_channels=2 * out_channels)
        )

    def forward(self, x):
        return self.module(x)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, n_block_1, n_block_2=1, use_maxpool=True):
        super().__init__()

        self.block = nn.Sequential(Module(in_channels, out_channels))

        for i in range(n_block_1 - 1):
            self.block.append(Module(in_channels, out_channels))
        
        if n_block_2 > 0:
            self.block.append(Module(2 * out_channels, 2 * out_channels))
        
        if use_maxpool:
            self.block.append(nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ))

    def forward(self, x):
        return self.block(x)


class YOLO(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_1 = nn.Sequential(
            Conv(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            Conv(
                in_channels=64,
                out_channels=192
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            Block(
                in_channels=192,
                out_channels=128,
                n_block_1=1
            ),
            Block(
                in_channels=512,
                out_channels=256,
                n_block_1=4
            ),
            Block(
                in_channels=1024,
                out_channels=512,
                n_block_1=2,
                n_block_2=0,
                use_maxpool=False
            )
        )

        n_channels = 1024

        self.feature_2 = nn.Sequential(
            Conv(
                in_channels=n_channels,
                out_channels=n_channels
            ),
            Conv(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=3,
                stride=2
            ),
            Conv(
                in_channels=n_channels,
                out_channels=n_channels
            ),
            Conv(
                in_channels=n_channels,
                out_channels=n_channels
            )
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50176, 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, 1470),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.feature_1(x)
        y = self.feature_2(y)
        return self.fc(y)

# test


class YOLO(torch.nn.Module):
    def __init__(self, VGG16):
        super(YOLO, self).__init__()
        self.backbone = VGG16

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024,
                      kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024,
                      kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=1024, out_channels=1024,
                      kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024,
                      kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(7*7*1024, 4096),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1470)
        )
    
    def forward(self, x):
        out = self.backbone(x)
        out = self.conv(out)
        out = self.linear(out)
        out = torch.reshape(out, (-1, 7, 7, 30))
        return out
