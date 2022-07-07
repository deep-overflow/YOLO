import torch
import math
import os
import numpy as np
from PIL import Image, ImageDraw
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from matplotlib import pyplot as plt

# math, plotting
import matplotlib.pyplot as plt
from math import sqrt
import random


# 픽셀값으로 되어 있는 bbox 값을 (이미지 크기에 대한) 상대값으로 변경해줍니다
def bbox_transform(bbox: dict, img_height, img_width) -> dict:
    bbox = bbox.copy()
    # 기존 값을 변경
    bbox['xmin'] = int(bbox['xmin']) / img_width        # must be float
    bbox['xmax'] = int(bbox['xmax']) / img_width
    bbox['ymin'] = int(bbox['ymin']) / img_height
    bbox['ymax'] = int(bbox['ymax']) / img_height
    bbox['x'] = (bbox['xmin'] + bbox['xmax']) / 2
    bbox['y'] = (bbox['ymin'] + bbox['ymax']) / 2
    bbox['w'] = bbox['xmax'] - bbox['xmin']
    bbox['h'] = bbox['ymax'] - bbox['ymin']
    # 새 값 추가
    bbox['sqrt_w'] = sqrt(bbox['w'])
    bbox['sqrt_h'] = sqrt(bbox['h'])

    return bbox

def print_img(img, annotation):
    """  
    {'annotation': {'filename': '000017.jpg',
                'folder': 'VOC2007',
                'object': [{'bndbox': {'xmax': '279',
                                       'xmin': '185',
                                       'ymax': '199',
                                       'ymin': '62'},
                            'difficult': '0',
                            'name': 'person',
                            'pose': 'Left',
                            'truncated': '0'},
                           {'bndbox': {'xmax': '403',
                                       'xmin': '90',
                                       'ymax': '336',
                                       'ymin': '78'},
                            'difficult': '0',
                            'name': 'horse',
                            'pose': 'Left',
                            'truncated': '0'}],
                'owner': {'flickrid': 'genewolf', 'name': 'whiskey kitten'},
                'segmented': '0',
                'size': {'depth': '3', 'height': '364', 'width': '480'},
                'source': {'annotation': 'PASCAL VOC2007',
                           'database': 'The VOC2007 Database',
                           'flickrid': '228217974',
                           'image': 'flickr'}}}
    """

    # 1. RGB로 바꾸어주기
    img = img.convert('RGB')

    # 2. 사각형 그리기
    # (xmin, ymin, xmax, ymax)
    draw = ImageDraw.Draw(img)
    for object in annotation['annotation']['object']:
        box = (float(object['bndbox']['xmin']), float(object['bndbox']['ymin']), float(object['bndbox']['xmax']),  float(object['bndbox']['ymax']))
        color = tuple(np.random.randint(low= 0, high= 256, size=3))
        width = 3
        text_pos = ((box[0]) + width, box[1])

        draw.rectangle(box, outline=color, width=width)
        draw.text(text_pos, object['name'], color = color)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img)
    

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
# (x,y,w,h) tensor 둘을 받아서 IoU를 계산합니다
def IOU(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    
    assert len(bbox1) == 4, 'bbox1이 이상합니다'
    assert len(bbox2) == 4, 'bbox2가 이상합니다'

    
    max_x1 = bbox1[0] + (bbox1[2] / 2)
    min_x1 = bbox1[0] - (bbox1[2] / 2)
    max_y1 = bbox1[1] + (bbox1[3] / 2)
    min_y1 = bbox1[1] - (bbox1[3] / 2)

    max_x2 = bbox2[0] + (bbox2[2] / 2)
    min_x2 = bbox2[0] - (bbox2[2] / 2)
    max_y2 = bbox2[1] + (bbox2[3] / 2)
    min_y2 = bbox2[1] - (bbox2[3] / 2)

    # 직사각형 A, B의 넓이를 구한다.
    # get area of rectangle A and B
    rect1_area = (max_x1 - min_x1) * (max_y1 - min_y1)
    rect2_area = (max_x2 - min_x2) * (max_y2 - min_y2)

    # Intersection의 가로와 세로 길이를 구한다.
    # get length and width of intersection.
    intersection_x_length = min(max_x1, max_x2) - max(min_x1, min_x2)
    intersection_y_length = min(max_y1, max_y2) - max(min_y1, min_y2)
    
    
    # width와 length의 길이가 유효하다면 IoU를 구한다.
    # If the width and length are valid, get IoU.
    if (bool(intersection_x_length > 0) & bool(intersection_y_length > 0)):
        intersection_area = intersection_x_length * intersection_y_length
        union_area = rect1_area + rect2_area - intersection_area
        ret = intersection_area / union_area
    else :
        ret = 0
    return torch.Tensor([ret])

""" 
코드 출처
https://gaussian37.github.io/math-algorithm-iou/
"""


