

def yolo_multitask_loss(y_pred, y_true):  # 커스텀 손실함수. 배치 단위로 값이 들어온다

    batch_loss = 0
    count = len(y_true)
    for i in range(0, len(y_true)):
        y_true_unit = y_true[i].clone().detach().requires_grad_(True)

        y_pred_unit = y_pred[i].clone().detach().requires_grad_(True)

        y_true_unit = torch.reshape(y_true_unit, [49, 25])
        y_pred_unit = torch.reshape(y_pred_unit, [49, 30])

        loss = 0

        for j in range(0, len(y_true_unit)):
            # pred = [1, 30], true = [1, 25]

            bbox1_pred = y_pred_unit[j, :4].clone(
            ).detach().requires_grad_(True)
            bbox1_pred_confidence = y_pred_unit[j, 4].clone(
            ).detach().requires_grad_(True)
            bbox2_pred = y_pred_unit[j, 5:9].clone(
            ).detach().requires_grad_(True)
            bbox2_pred_confidence = y_pred_unit[j, 9].clone(
            ).detach().requires_grad_(True)
            class_pred = y_pred_unit[j, 10:].clone(
            ).detach().requires_grad_(True)

            bbox_true = y_true_unit[j, :4].clone(
            ).detach().requires_grad_(True)
            bbox_true_confidence = y_true_unit[j, 4].clone(
            ).detach().requires_grad_(True)
            class_true = y_true_unit[j, 5:].clone(
            ).detach().requires_grad_(True)
            
            # IoU 구하기
            # x,y,w,h -> min_x, min_y, max_x, max_y로 변환
            box_pred_1_np = bbox1_pred.detach().numpy()
            box_pred_2_np = bbox2_pred.detach().numpy()
            box_true_np = bbox_true.detach().numpy()

            box_pred_1_area = box_pred_1_np[2] * box_pred_1_np[3]
            box_pred_2_area = box_pred_2_np[2] * box_pred_2_np[3]
            box_true_area = box_true_np[2] * box_true_np[3]

            box_pred_1_minmax = np.asarray([box_pred_1_np[0] - 0.5*box_pred_1_np[2], box_pred_1_np[1] - 0.5 *
                                           box_pred_1_np[3], box_pred_1_np[0] + 0.5*box_pred_1_np[2], box_pred_1_np[1] + 0.5*box_pred_1_np[3]])
            box_pred_2_minmax = np.asarray([box_pred_2_np[0] - 0.5*box_pred_2_np[2], box_pred_2_np[1] - 0.5 *
                                           box_pred_2_np[3], box_pred_2_np[0] + 0.5*box_pred_2_np[2], box_pred_2_np[1] + 0.5*box_pred_2_np[3]])
            box_true_minmax = np.asarray([box_true_np[0] - 0.5*box_true_np[2], box_true_np[1] - 0.5 *
                                         box_true_np[3], box_true_np[0] + 0.5*box_true_np[2], box_true_np[1] + 0.5*box_true_np[3]])

            # 곂치는 영역의 (min_x, min_y, max_x, max_y)
            InterSection_pred_1_with_true = [max(box_pred_1_minmax[0], box_true_minmax[0]), max(box_pred_1_minmax[1], box_true_minmax[1]), min(
                box_pred_1_minmax[2], box_true_minmax[2]), min(box_pred_1_minmax[3], box_true_minmax[3])]
            InterSection_pred_2_with_true = [max(box_pred_2_minmax[0], box_true_minmax[0]), max(box_pred_2_minmax[1], box_true_minmax[1]), min(
                box_pred_2_minmax[2], box_true_minmax[2]), min(box_pred_2_minmax[3], box_true_minmax[3])]

            # 박스별로 IoU를 구한다
            IntersectionArea_pred_1_true = 0

            # 음수 * 음수 = 양수일 수도 있으니 검사를 한다.
            if (InterSection_pred_1_with_true[2] - InterSection_pred_1_with_true[0] + 1) >= 0 and (InterSection_pred_1_with_true[3] - InterSection_pred_1_with_true[1] + 1) >= 0:
                IntersectionArea_pred_1_true = (
                    InterSection_pred_1_with_true[2] - InterSection_pred_1_with_true[0] + 1) * InterSection_pred_1_with_true[3] - InterSection_pred_1_with_true[1] + 1

            IntersectionArea_pred_2_true = 0

            if (InterSection_pred_2_with_true[2] - InterSection_pred_2_with_true[0] + 1) >= 0 and (InterSection_pred_2_with_true[3] - InterSection_pred_2_with_true[1] + 1) >= 0:
                IntersectionArea_pred_2_true = (
                    InterSection_pred_2_with_true[2] - InterSection_pred_2_with_true[0] + 1) * InterSection_pred_2_with_true[3] - InterSection_pred_2_with_true[1] + 1

            Union_pred_1_true = box_pred_1_area + \
                box_true_area - IntersectionArea_pred_1_true
            Union_pred_2_true = box_pred_2_area + \
                box_true_area - IntersectionArea_pred_2_true

            IoU_box_1 = IntersectionArea_pred_1_true/Union_pred_1_true
            IoU_box_2 = IntersectionArea_pred_2_true/Union_pred_2_true

            responsible_box = 0
            responsible_bbox_confidence = 0
            non_responsible_bbox_confidence = 0

            # box1, box2 중 responsible한걸 선택(IoU 기준)
            if IoU_box_1 >= IoU_box_2:
                responsible_box = bbox1_pred.clone().detach().requires_grad_(True)
                responsible_bbox_confidence = bbox1_pred_confidence.clone().detach().requires_grad_(True)
                non_responsible_bbox_confidence = bbox2_pred_confidence.clone(
                ).detach().requires_grad_(True)

            else:
                responsible_box = bbox2_pred.clone().detach().requires_grad_(True)
                responsible_bbox_confidence = bbox2_pred_confidence.clone().detach().requires_grad_(True)
                non_responsible_bbox_confidence = bbox1_pred_confidence.clone(
                ).detach().requires_grad_(True)
