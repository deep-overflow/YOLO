"""  

"""

import torch
from math import sqrt
import numpy as np
from functions import IOU
from datasets import get_name_labels_dict

def loss_func(config, pred : torch.Tensor, label_list : list):
    """  
    Loss를 계산합니다.

    Args:
        label_list (list): a list of dictionary
        pred (torch.Tensor): (batch, S, S, 5*B + C) = (batch, 7, 7, 30)
    """
    
    # ============ Loss 계산 ===========

    mse = torch.nn.MSELoss(reduction='sum')
    loss_list = list()

    # 각 batch 중, 하나의 데이터마다 각각 loss를 계산한다
    for batch, label in enumerate(label_list):
        """  
        Args
            label (dict) 
                e.g. {'annotation': {'filename': '000012.jpg',
                        'folder': 'VOC2007',
                        'object': [{'bbox': {'h': 0.5195195195195196,
                                            'sqrt_h': 0.7207770248277338,
                                            'sqrt_w': 0.6244997998398398,
                                            'w': 0.38999999999999996,
                                            'x': 0.507,
                                            'xmax': 0.702,
                                            'xmin': 0.312,
                                            'y': 0.5510510510510511,
                                            'ymax': 0.8108108108108109,
                                            'ymin': 0.2912912912912913},
                                    'name': 'car'}],
                        'owner': {'flickrid': 'KevBow', 'name': '?'},
                        'segmented': '0',
                        'size': {'depth': '3', 'height': '333', 'width': '500'},
                        'source': {'annotation': 'PASCAL VOC2007',
                                'database': 'The VOC2007 Database',
                                'flickrid': '207539885',
                                'image': 'flickr'}}}
        """

        img_height = config.MODEL.HEIGHT
        img_width = config.MODEL.WIDTH

        # 각 이미지에 포함되어 있는 object 정보들에 대한 list
        obj_list = label['annotation']['object']
        num_obj = len(obj_list)

        # 순회를 체크하기 위해 S X S Grid를 만든다.
        # 기본값은 0
        # object가 있는 cell은 1로 표시한다
        GRID = np.zeros((config.MODEL.S, config.MODEL.S))

        # ============= Object가 존재하는 Cell의 Loss를 먼저 계산한다 =================
        for n in range(num_obj):

            """
            Example of obj_info:
            {'bbox': {'h': 0.93048128342246,
                    'sqrt_h': 0.9646145776539249,
                    'sqrt_w': 0.5709640969448079,
                    'w': 0.326,
                    'x': 0.20299999999999999,
                    'xmax': 0.366,
                    'xmin': 0.04,
                    'y': 0.4839572192513369,
                    'ymax': 0.9491978609625669,
                    'ymin': 0.01871657754010695},
            'name': 'person'}
            """
            obj_info = obj_list[n]

            # (i,j) cell에 떨어진다고 할 때, (i,j)를 찾는다
            x = obj_info['bbox']['x']
            y = obj_info['bbox']['y']
            i = int(y * config.MODEL.S)
            j = int(x * config.MODEL.S)

            # 이미 해당 cell에 object가 있었다면 pass
            # YOLO는 한 cell에서 단 하나의 object만을 탐지하기 때문이다
            if (GRID[i][j] == 1):
                continue
            else:
                GRID[i][j] = 1

            # classification loss
            pred_probs = pred[batch, i, j, 5*config.MODEL.B:]           # (C,) tensor
            true_label = get_name_labels_dict()[obj_info['name']]      # Answer Label e.g.) 4
            label_probs = torch.zeros_like(pred_probs)
            label_probs[true_label] = 1.0
            loss_list.append(mse(pred_probs, label_probs))


            # Responsible한 bbox를 찾는다 : target bbox와의 IOU가 제일 큰 bbox
            max_IOU = torch.Tensor([-1]).to('cpu')
            reponsible_bbox_num = -1

            # num_bbox 0 ~ B-1
            for num_bbox in range(config.MODEL.B):
                pred_coord = pred[batch, i, j, 5 * (num_bbox): 5*(num_bbox) + 4]
                target_coord = torch.Tensor(
                    [obj_info['bbox']['x'], obj_info['bbox']['y'], obj_info['bbox']['w'], obj_info['bbox']['h']]).to('cpu')
                iou = IOU(target_coord, pred_coord).to('cpu')
                if bool(iou > max_IOU):
                    reponsible_bbox_num = num_bbox
                    max_IOU = iou

            # print('max_IOU : ', max_IOU)

            for num_bbox in range(config.MODEL.B):
                pred_coord = pred[batch, i, j, 5 * (num_bbox): 5*(num_bbox) + 4]        # (4,) tensor
                pred_confidence = pred[batch, i, j, 5*(num_bbox) + 4].unsqueeze(0)                   # (1,) tensor
                # print('pred_confidence : ', pred_confidence.item())

                # responsible한 bbox의 경우
                if (num_bbox == reponsible_bbox_num):
                    sqrt_pred_coord = torch.zeros_like(pred_coord)
                    sqrt_pred_coord[0] = pred_coord[0]
                    sqrt_pred_coord[1] = pred_coord[1]
                    sqrt_pred_coord[2] = sqrt(pred_coord[2])     # w -> sqrt(w)
                    sqrt_pred_coord[3] = sqrt(pred_coord[3])     # h -> sqrt(h)

                    target_coord = torch.Tensor(
                        [obj_info['bbox']['x'], obj_info['bbox']['y'], obj_info['bbox']['sqrt_w'], obj_info['bbox']['sqrt_h']])
                    target_coord = target_coord.to('cpu')

                    # localization(coordinate) loss
                    assert pred_coord.size() == target_coord.size()
                    loss_list.append(config.LOSS.LAMBDA_COORD * \
                        mse(sqrt_pred_coord, target_coord))
                    # confidence error loss
                    assert pred_confidence.size() == max_IOU.size()
                    loss_list.append(mse(pred_confidence, max_IOU))

                # reponsible하지 않은 bbox의 경우
                else:
                    # If no object exists in cell, the  confidence scores should be zero
                    loss_list.append(config.LOSS.LAMBDA_NOOBJ * mse(pred_confidence, torch.Tensor([0]).to('cpu')))

        # =========== Object가 존재하지 않는 Cell의 Loss ===========

        for i in range(config.MODEL.S):
            for j in range(config.MODEL.S):
                # 이미 loss를 계산했던 Cell인지 확인
                if (GRID[i][j] == 1):
                    continue
                else:
                    GRID[i][j] = 1

                for num_bbox in range(config.MODEL.B):
                    pred_confidence = pred[batch, i, j, 5*(num_bbox) + 4].unsqueeze(0)
                    loss_list.append(config.LOSS.LAMBDA_NOOBJ * \
                        mse(pred_confidence, torch.Tensor([0]).to('cpu')))

    # print('loss_list : ', loss_list)
    loss = torch.sum(torch.stack(loss_list))  / len(label_list)
    
    return loss
                    
