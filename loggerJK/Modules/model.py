import timm
from einops import rearrange, reduce, repeat
from torchsummary import summary

import torch
import torch.nn as nn
import torchvision.datasets as transforms

""" 
input : (3, 448, 448) = (C, H, W)
output : S * S * (B * 5 + C)
"""
class Yolo(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 모델 상수
        self.S = config.MODEL.S
        self.B = config.MODEL.B
        self.C = config.MODEL.C
        
        # backbone
        self.backbone = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=True)
        self.backbone.reset_classifier(self.S * self.S * (5 * self.B + self.C))
        self.sigmoid = nn.Sigmoid()

        # freeze backbone
        freeze_exception = ['norm.weight',
                            'norm.bias',
                            'head.weight',
                            'head.bias']

        for name, param in self.backbone.named_parameters():
            if (name not in freeze_exception):
                # print(name)
                param.requires_grad = False




    def forward(self, x):
        out = self.backbone(x)
        out = self.sigmoid(out).clone()
        # out = rearrange(out, 'bs (S s X) -> bs S s X', S = self.S, s = self.S)
        out = torch.reshape(out, (-1, self.S, self.S, 5 * self.B + self.C))

        return out
    
def build_model(config):
    model = Yolo(config)
    return model
