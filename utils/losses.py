
import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_chose(loss_func):
    if loss_func == 'SoftIoULoss':
        cirterion = SoftIoULoss()
    elif loss_func == 'BCEWithLogitsLoss':
        cirterion = nn.BCEWithLogitsLoss(size_average=False)
    elif loss_func == 'BCELoss':
        cirterion = nn.BCELoss(size_average=True)
    elif loss_func == 'ISNetLoss':
        cirterion = ISNetLoss()
    else:
        raise ('An unexpected loss function!')

    return cirterion


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks):
        edge_gt = self.grad(gt_masks.clone())
        output, edge_out = preds

        loss_img = self.softiou(output, gt_masks)
        loss_edge = 10 * self.bce(edge_out, edge_gt)+ self.softiou(edge_out, edge_gt)
        
        return loss_img + loss_edge

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0

class DPLoss(nn.Module):
    def __init__(self, loss):
        super(DPLoss, self).__init__()
        self.loss = loss
        
    def forward(self, preds, gt_masks):
        if isinstance(preds, list) or isinstance(preds, tuple):
            a = []
            for i in range(len(preds)):
                pred = preds[i]
                loss = self.loss(pred, gt_masks)
                a.append(loss)
            loss_total = sum(a)
            return loss_total

        else:
            loss = self.loss(preds, gt_masks)
            return loss
