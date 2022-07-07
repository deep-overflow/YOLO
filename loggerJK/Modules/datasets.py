"""  
TODO
- collate_fn() config 어떻게 넘겨줄지 고민
"""
import torch
import torch.nn as nn
import torchvision.datasets as transforms
from torchvision.datasets import VOCDetection
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from PIL import Image, ImageDraw

from functions import *


from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
import collections
from typing import Any, Callable, Dict, Optional, Tuple, List

labels_list = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane',
        'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle',
            'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

labels_name_dict = dict()
name_labels_dict = dict()
for i, label in enumerate(labels_list):
    labels_name_dict[i] = label
    name_labels_dict[label] = i
    
def get_name_labels_dict() -> dict:
    return name_labels_dict

class CustomCollateFn():
    def __init__ (self, config):
        self.config = config

    def __call__ (self, datas: list):
        """ 
        1. 이미지들의 크기를 조정하고, 배치 단위 텐서로 변환합니다.
        2. list of label dictionary를 반환합니다.
        Args :
            datas : is a list of tuple (image, dictionary)

        Return :
            batch_img (torch.Tensor) : (B, C, H, W)
            label_list (list) : a list of dictionary
        """
        img_list = list()
        label_list = list()

        for i in range(len(datas)):
            img = np.array(datas[i][0])   # PIL Image -> numpy array, (H, W, C)
            original_img_height = img.shape[0]
            original_img_width = img.shape[1]
            label = datas[i][1]     # dict

            # label['original_img_height'] = original_img_height
            # label['original_img_width'] = original_img_width

            # numpy -> tensor
            # TODO : config 처리
            transform = A.Compose([
                A.Resize(self.config.MODEL.HEIGHT, self.config.MODEL.WIDTH),
                ToTensorV2(),
            ])
            img_tensor = transform(image=img)['image'].to(dtype=torch.float)    # (C, H, W) Tensor

            # 각 이미지에 포함되어 있는 object 정보들에 대한 list
            original_obj_list = label['annotation']['object']

            """  
            데이터를 가공합니다.
            bbox 값을 (이미지 크기에 대한) 상대값으로 설정

            obj_list (list) : a list of dictionary
                e.g. obj_list[1]['name'] : 1번째 object의 name
            """
            obj_list = list()
            for obj in original_obj_list:
                ratio_bbox = bbox_transform(
                    bbox=obj['bndbox'], img_height=original_img_height, img_width=original_img_width)

                obj_info = dict()
                obj_info['name'] = obj['name']
                obj_info['bbox'] = ratio_bbox

                obj_list.append(obj_info)
            
            label['annotation']['object'] = obj_list

            img_list.append(img_tensor)
            label_list.append(label)


        batch_img = torch.stack(img_list, dim=0)

        return batch_img, label_list
        
        


# def custom_collate_fn(datas : list):
#     """ 
#     1. 이미지들의 크기를 조정하고, 배치 단위 텐서로 변환합니다.
#     2. list of label dictionary를 반환합니다.
#     Args :
#         datas : is a list of tuple (image, dictionary)

#     Return :
#         batch_img (torch.Tensor) : (B, C, H, W)
#         label_list (list) : a list of dictionary
#     """
#     img_list = list()
#     label_list = list()

#     for i in range(len(datas)):
#         img = np.array(datas[i][0])   # PIL Image -> numpy array, (H, W, C)
#         original_img_height = img.shape[0]
#         original_img_width = img.shape[1]
#         label = datas[i][1]     # dict

#         # label['original_img_height'] = original_img_height
#         # label['original_img_width'] = original_img_width

#         # numpy -> tensor
#         transform = A.Compose([
#             A.Resize(config.MODEL.HEIGHT, config.MODEL.WIDTH),
#             ToTensorV2(),
#         ])
#         img_tensor = transform(image=img)['image'].to(dtype=torch.float)    # (C, H, W) Tensor

#         # 각 이미지에 포함되어 있는 object 정보들에 대한 list
#         original_obj_list = label['annotation']['object']

#         """  
#         데이터를 가공합니다.
#         bbox 값을 (이미지 크기에 대한) 상대값으로 설정

#         obj_list (list) : a list of dictionary
#             e.g. obj_list[1]['name'] : 1번째 object의 name
#         """
#         obj_list = list()
#         for obj in original_obj_list:
#             ratio_bbox = bbox_transform(
#                 bbox=obj['bndbox'], img_height=original_img_height, img_width=original_img_width)

#             obj_info = dict()
#             obj_info['name'] = obj['name']
#             obj_info['bbox'] = ratio_bbox

#             obj_list.append(obj_info)
        
#         label['annotation']['object'] = obj_list

#         img_list.append(img_tensor)
#         label_list.append(label)


#     batch_img = torch.stack(img_list, dim=0)

#     return batch_img, label_list



def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
    # 리턴값은 일반 Dict
    voc_dict: Dict[str, Any] = {}

    # children : [자식element1, 자식element2, 자식element3, ...]
    children = list(node)
    if children:
        # 중복된 태그들은 하나의 list로 관리한다
        # e.g.) {object : [{ }, { }, { }, ... ] }
        def_dic: Dict[str, Any] = collections.defaultdict(list)
        for dc in map(VOCDetection.parse_voc_xml, children):
            for ind, v in dc.items():
                def_dic[ind].append(v)
        if node.tag == "annotation":
            def_dic["object"] = [def_dic["object"]]
        voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
    # element에 text가 있는 경우
    if node.text:
        # text를 얻는다
        text = node.text.strip()
        # 자식이 없는경우
        if not children:
            # {tag : text}를 voc_dict에 추가한다
            voc_dict[node.tag] = text
    return voc_dict

    
class customVOCdataset(Dataset):
    """  
    Train 커스텀 데이터셋
    """
    def __init__(self, root:str, mode:str):
        self.root = root
        if (mode == 'train'):
            self.img_list_path = Path(root) / Path('VOCdevkit/VOC2007/ImageSets/Main/train.txt')
        elif (mode == 'val'):
            self.img_list_path = Path(root) / Path('VOCdevkit/VOC2007/ImageSets/Main/val.txt')
        elif (mode == 'trainval'):
            self.img_list_path = Path(root) / Path('VOCdevkit/VOC2007/ImageSets/Main/trainval.txt')
        elif (mode == 'test'):
            self.img_list_path = Path(root) / Path('VOCdevkit/VOC2007/ImageSets/Main/test.txt')
        self.img_list = list()
        with open(self.img_list_path) as f:
            for line in f:
                self.img_list.append(line.strip())

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index : int):    
        """  
        Return :
        - img (PIL image)
        - annotation (dict)
        """

        self.img_file_name = self.img_list[index]
        self.img_path = Path(self.root) / Path('VOCdevkit/VOC2007/JPEGImages') / Path(self.img_file_name + '.jpg')
        annotation_path = Path(self.root) / Path('VOCdevkit/VOC2007/Annotations') / Path(self.img_file_name + '.xml')

        annotation_str =''
        with open(annotation_path) as f:
            for line in f:
                annotation_str += line

        # pprint(annotation_str)
        
        img = Image.open(self.img_path)
        target = parse_voc_xml(ET_parse(annotation_path).getroot())

        return (img, target)
    

    
    
def build_dataset(config):
    root_path = config.DATASET.PATH
    downlaod = VOCDetection(root=root_path, year='2007',image_set='train', download=True,)
    
    custom_collate_fn = CustomCollateFn(config)

    train_dataset = customVOCdataset(root=root_path, mode = 'train')
    val_dataset = customVOCdataset(root=root_path, mode = 'val')
    trainval_dataset = customVOCdataset(root=root_path, mode = 'trainval')
    test_dataset = customVOCdataset(root=root_path, mode = 'test')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.MODEL.BATCH_SIZE, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.MODEL.BATCH_SIZE, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size=config.MODEL.BATCH_SIZE, collate_fn=custom_collate_fn)
    trainval_dataloader = DataLoader(dataset = trainval_dataset, batch_size=config.MODEL.BATCH_SIZE, collate_fn=custom_collate_fn)


    return train_dataset, val_dataset, test_dataset, trainval_dataset, train_dataloader, val_dataloader, test_dataloader, trainval_dataloader