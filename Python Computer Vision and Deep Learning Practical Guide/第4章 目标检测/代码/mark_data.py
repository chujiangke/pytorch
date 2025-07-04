import torch
from torch.nn import CrossEntropyLoss, L1Loss

from config import img_size, device


def mark_data(boxes):
    label_matrix = torch.zeros((3, 3)).to(device)
    offset_matrix = torch.ones((3, 3, 3)).to(device)
    confidences = torch.zeros((3, 3)).to(device)
    # 格子尺寸
    grid_w = grid_h = img_size / 3
    grids = torch.Tensor(
        [
            [100, 100, 100],
            [200, 100, 100],
            [300, 100, 100],
            [100, 200, 100],
            [200, 200, 100],
            [300, 200, 100],
            [100, 300, 100],
            [200, 300, 100],
            [300, 300, 100],
        ]
    )
    for box in boxes:
        cx, cy, w = box
        h = w
        # 物体所在格子的编号
        grid_x = int(cx / grid_w)
        grid_y = int(cy / grid_h)
        label_matrix[grid_y, grid_x] = 1
        # cx,cy 均以格子右下角坐标计算offset
        # w以整个图片计算offset，以保证所有数值都在0-1之间
        offset_matrix[grid_y, grid_x] = torch.Tensor(
            [
                cx / ((grid_x * grid_w + grid_w)),
                cy / ((grid_y * grid_h + grid_h)),
                w / (img_size),
            ]
        )
        # box与grid的iou
        grid_box = grids[grid_x + 3 * grid_y]
        confidences[grid_y, grid_x] = iou(box, grid_box)

    return (
        label_matrix.view(-1, 9),
        offset_matrix.view(-1, 9, 3),
        confidences.view(-1, 9),
    )


class multi_box_loss(torch.nn.Module):
    def forward(
        self,
        label_prediction,
        offset_prediction,
        confidence_prediction,
        boxes_list,
    ):

        # def multi_box_loss(label_prediction,offset_prediction,boxes_list):
        # boxes_list 多张图片中的boxes列表
        reg_criteron = L1Loss()
        label_tensor = []
        offset_tensor = []
        confidence_tensor = []

        for boxes in boxes_list:
            label, offset, confidence = mark_data(boxes)
            label_tensor.append(label)
            offset_tensor.append(offset)
            confidence_tensor.append(confidence)
        label_tensor = torch.cat(label_tensor, dim=0).long()
        offset_tensor = torch.cat(offset_tensor, dim=0)
        confidence_tensor = torch.cat(confidence_tensor, dim=0)
        # 添加掩码，负例不加入回归计算
        mask = label_tensor == 1
        mask = mask.unsqueeze(2).float()
        label_prediction = label_prediction.permute(0, 2, 1)
        # weight = (label_tensor != 1) * 0.5
        # print(weight)
        weight = torch.Tensor([0.5, 1.5]).to(device)
        cls_criteron = CrossEntropyLoss(weight=weight.float())
        cls_loss = cls_criteron(label_prediction, label_tensor)
        offset_prediction = offset_prediction.view(-1, 9, 3)
        reg_loss = reg_criteron(offset_prediction * mask, offset_tensor * mask)
        # 转换mask维度，以便与confidence相乘
        mask = mask.squeeze(2)
        confidence_loss = reg_criteron(
            confidence_prediction * mask, confidence_tensor * mask
        )
        return cls_loss + reg_loss + confidence_loss


def iou(box1, box2):
    # box: cx,cy,w 正方形
    # box1
    cx_1, cy_1, w_1 = box1[:3]
    xmin_1 = cx_1 - w_1 / 2
    ymin_1 = cy_1 - w_1 / 2
    xmax_1 = cx_1 + w_1 / 2
    ymax_1 = cy_1 + w_1 / 2
    # box2
    cx_2, cy_2, w_2 = box2[:3]
    xmin_2 = cx_2 - w_2 / 2
    ymin_2 = cy_2 - w_2 / 2
    xmax_2 = cx_2 + w_2 / 2
    ymax_2 = cy_2 + w_2 / 2

    # 没有重叠则iou = 0
    if (
        ymax_1 <= ymin_2
        or ymax_2 <= ymin_1
        or xmax_2 <= xmin_1
        or xmax_1 <= xmin_2
    ):
        return 0.0

    inter_x_min = max(xmin_1, xmin_2)
    inter_y_min = max(ymin_1, ymin_2)
    inter_x_max = min(xmax_1, xmax_2)
    inter_y_max = min(ymax_1, ymax_2)

    intersection = (inter_y_max - inter_y_min) * (inter_x_max - inter_x_min)
    return intersection / (w_1 * w_1 + w_2 * w_2)

