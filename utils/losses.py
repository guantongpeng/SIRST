import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_chose(args):
    MyWgt = torch.Tensor(args.MyWgt)

    if args.loss_func == 'fullySup':
        cirterion = SoftIoULoss()
    elif args.loss_func == 'fullySup1':
        cirterion = SoftLoULoss1()
    elif args.loss_func == 'fullySup2':
        cirterion = muti_bce_loss_fusion()
    elif args.loss_func == 'fullySupBCE':
        cirterion = nn.BCEWithLogitsLoss(size_average=False)
    elif args.loss_func == 'FocalLoss':
        cirterion = Focal_Loss(alpha=MyWgt, gamma=2)
    elif args.loss_func == 'OHEM':
        cirterion = MyWeightTopKLoss_Absolutly(
            alpha=MyWgt,
            gamma=2,
            MaxClutterNum=args.MaxClutterNum,
            ProtectedArea=args.ProtectedArea)
    elif args.loss_func == 'HPM':
        cirterion = MyWeightBCETopKLoss(alpha=MyWgt,
                                        gamma=2,
                                        MaxClutterNum=args.MaxClutterNum,
                                        ProtectedArea=args.ProtectedArea)
    else:
        raise ('An unexpected loss function!')

    return cirterion


class Focal_Loss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha(0), alpha(1)])
        # if isinstance(alpha, (float, int)): self.alpha = torch.Tensor(alpha)
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        if input.dim() > 4:
            input = torch.squeeze(input, 2)
        logpt = F.logsigmoid(input)
        logpt_bk = F.logsigmoid(-input)
        pt = logpt.data.exp()
        pt_bk = logpt_bk.data.exp()
        loss = -self.alpha[1] * (
            1 - pt)**self.gamma * target * logpt - self.alpha[
                0] * pt_bk**self.gamma * (1 - target) * logpt_bk

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

    
class SoftIoULoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=False):
        super(SoftIoULoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha(0), alpha(1)])
        # if isinstance(alpha, (float, int)): self.alpha = torch.Tensor(alpha)
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)

    def forward(self, pred, target):
        if pred.dim() > 4:
            pred = torch.squeeze(pred, 2)

        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss


class SoftLoULoss1(nn.Module):

    def __init__(self, batch=32):
        super(SoftLoULoss1, self).__init__()
        self.batch = batch
        # self.bce_loss = nn.BCELoss()

    def forward(self, pred, target):
        if pred.dim() > 4:
            pred = torch.squeeze(pred, 2)

        pred = torch.sigmoid(pred)
        smooth = 0.00

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))
        loss = (intersection_sum + smooth) / (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)
        # loss1 = self.bce_loss(pred, target)
        return loss


class muti_SoftLoULoss1_fusion(nn.Module):

    def __init__(self, size_average=True):
        super(muti_SoftLoULoss1_fusion, self).__init__()

        self.softIou = SoftLoULoss1()

    def forward(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        if d0.dim() > 4:
            d0 = torch.squeeze(d0, 2)

        loss0 = self.softIou(d0, labels_v)
        loss1 = self.softIou(d1, labels_v)
        loss2 = self.softIou(d2, labels_v)
        loss3 = self.softIou(d3, labels_v)
        loss4 = self.softIou(d4, labels_v)
        loss5 = self.softIou(d5, labels_v)
        loss6 = self.softIou(d6, labels_v)

        loss = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6) / 7

        return loss0, loss


class muti_bce_loss_fusion(nn.Module):

    def __init__(self, size_average=True):
        super(muti_bce_loss_fusion, self).__init__()

        self.bce_loss = nn.BCELoss(size_average=size_average)

    def forward(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        if d0.dim() > 4:
            d0 = torch.squeeze(d0, 2)

        loss0 = self.bce_loss(torch.sigmoid(d0), labels_v)
        loss1 = self.bce_loss(torch.sigmoid(d1), labels_v)
        loss2 = self.bce_loss(torch.sigmoid(d2), labels_v)
        loss3 = self.bce_loss(torch.sigmoid(d3), labels_v)
        loss4 = self.bce_loss(torch.sigmoid(d4), labels_v)
        loss5 = self.bce_loss(torch.sigmoid(d5), labels_v)
        loss6 = self.bce_loss(torch.sigmoid(d6), labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        return loss0, loss


# class bce_loss_NPos(nn.Module):
#     def __init__(self):
#         super(bce_loss_NPos, self).__init__()
#
#         self.bce_loss = nn.BCELoss(reduce=False)
#
#     def forward(self, input, target):
#         input = torch.sigmoid(input)
#         losses = self.bce_loss(input, target)
#         positives = losses[target==1]
#
#         return loss

def Dice( pred, target,warm_epoch=1, epoch=1, layer=0):
        pred = torch.sigmoid(pred)
  
        smooth = 1

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))

        loss = (2*intersection_sum + smooth) / \
            (pred_sum + target_sum + intersection_sum + smooth)

        loss = 1 - loss.mean()

        return loss

class SLSIoULoss(nn.Module):
    def __init__(self):
        super(SLSIoULoss, self).__init__()


    def forward(self, pred_log, target,warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        dis = torch.pow((pred_sum-target_sum)/2, 2)
        
        
        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth) 
        
        loss = (intersection_sum + smooth) / \
                (pred_sum + target_sum - intersection_sum  + smooth)       
        lloss = LLoss(pred, target)

        if epoch>warm_epoch:       
            siou_loss = alpha * loss
            if with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 -siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss
    
    

def LLoss(pred, target):
        loss = torch.tensor(0.0, requires_grad=True).to(pred)

        patch_size = pred.shape[0]
        h = pred.shape[2]
        w = pred.shape[3]        
        x_index = torch.arange(0,w,1).view(1, 1, w).repeat((1,h,1)).to(pred) / w
        y_index = torch.arange(0,h,1).view(1, h, 1).repeat((1,1,w)).to(pred) / h
        smooth = 1e-8
        for i in range(patch_size):  

            pred_centerx = (x_index*pred[i]).mean()
            pred_centery = (y_index*pred[i]).mean()

            target_centerx = (x_index*target[i]).mean()
            target_centery = (y_index*target[i]).mean()
           
            angle_loss = (4 / (torch.pi**2) ) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth)) 
                                                            - torch.arctan((target_centery) / (target_centerx + smooth))))

            pred_length = torch.sqrt(pred_centerx*pred_centerx + pred_centery*pred_centery + smooth)
            target_length = torch.sqrt(target_centerx*target_centerx + target_centery*target_centery + smooth)
            
            length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)
        
            loss = loss + (1 - length_loss + angle_loss) / patch_size
        
        return loss
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

################################ BCETopKLoss ####################################################
class MyWeightBCETopKLoss(nn.Module):

    def __init__(self,
                 gamma=0,
                 alpha=None,
                 size_average=False,
                 MaxClutterNum=39,
                 ProtectedArea=2):
        super(MyWeightBCETopKLoss, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(reduce=False)

        self.HardRatio = 1 / 4
        self.HardNum = round(MaxClutterNum * self.HardRatio)
        self.EasyNum = MaxClutterNum - self.HardNum

        self.MaxClutterNum = MaxClutterNum
        self.ProtectedArea = ProtectedArea
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha(0), alpha(1)])
        # if isinstance(alpha, (float, int)): self.alpha = torch.Tensor(alpha)
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)

    def forward(self, input,
                target):  ## Input: [2,1,512,512]    Target: [2,1,512,512]

        if input.dim() > 4:
            input = torch.squeeze(input, 2)

        ## target surrounding = 2
        template = torch.ones(1, 1, 2 * self.ProtectedArea + 1,
                              2 * self.ProtectedArea + 1).to(
                                  input.device)  ## [1,1,5,5]
        target_prot = F.conv2d(target.float(),
                               template,
                               stride=1,
                               padding=self.ProtectedArea)  ## [2,1,512,512]
        target_prot = (target_prot > 0).float()

        with torch.no_grad():
            loss_wise = self.bce_loss(
                input,
                target.float())  ## learning based on result of loss computing
            loss_p = loss_wise * (1 - target_prot)
            idx = torch.randperm(130) + 20

            batch_l = loss_p.shape[0]
            Wgt = torch.zeros(batch_l, 1, 512, 512)
            for ls in range(batch_l):
                loss_ls = loss_p[ls, :, :, :].reshape(-1)
                loss_topk, indices = torch.topk(loss_ls, 200)
                indices_rand = indices[idx[
                    0:self.
                    HardNum]]  ## random select HardNum samples in top [20-150]
                idx_easy = torch.randperm(len(loss_ls))[0:self.EasyNum].to(
                    input.device
                )  ## random select EasyNum samples in all image
                indices_rand = torch.cat((indices_rand, idx_easy), 0)
                indices_rand_row = indices_rand // 512
                indices_rand_col = indices_rand % 512
                Wgt[ls, 0, indices_rand_row, indices_rand_col] = 1

            WgtData_New = Wgt.to(
                input.device) * (1 - target_prot) + target.float()
            WgtData_New[WgtData_New > 1] = 1

        logpt = F.logsigmoid(input)
        logpt_bk = F.logsigmoid(-input)
        pt = logpt.data.exp()
        pt_bk = 1 - logpt_bk.data.exp()
        loss = -self.alpha[1] * (
            1 - pt)**self.gamma * target * logpt - self.alpha[
                0] * pt_bk**self.gamma * (1 - target) * logpt_bk

        loss = loss * WgtData_New

        return loss.sum()

        # if input.dim()>2:
        #     input=input.view(input.size(0), input.size(1),-1)   # N,C,D,H,W=>N,C,D*H*W
        #     input=input.transpose(1,2)                          # N,C,D*H*W=>N, D*H*W, C
        #     input=input.contiguous().view(-1,input.size(2))     # N,D*H*W,C=>N*D*H*W, C
        #
        #     WgtData_New = WgtData_New.view(WgtData_New.size(0), WgtData_New.size(1), -1)    # N,C,D,H,W=>N,C,D*H*W
        #     WgtData_New = WgtData_New.transpose(1, 2)                               # N,C,D*H*W=>N, D*H*W,C
        #     WgtData_New = WgtData_New.contiguous().view(-1, WgtData_New.size(2))        # N,D*H*W,C=>N*D*H*W,C
        #
        # target = target.view(-1,1)     ## [2*1*512*512,1]     #N,D,H,W=>1,N*D*H*W
        # logpt = F.log_softmax(input, dim=1)   ## [2*1*512*512,2]
        # logpt = logpt.gather(1,target) ##  zhiding rank 2 target
        # logpt = logpt*WgtData_New        #weight  ## predit of concern 39+1
        # logpt = logpt.view(-1)           #possibility of target
        # pt=logpt.data.exp()
        #
        #
        # if self.alpha is not None:
        #     if self.alpha.type()!=input.data.type():
        #         self.alpha=self.alpha.type_as(input.data).to(input.device)
        #     at=self.alpha.gather(0,target.data.view(-1))
        #     logpt=logpt*at   ##at= alpha
        #
        # loss=-1*(1-pt)**self.gamma*logpt
################################ OHEM ####################################################
class MyWeightTopKLoss_Absolutly(nn.Module):
    def __init__(self,gamma=0, alpha=None, size_average=False, MaxClutterNum=39, ProtectedArea=2):
        super(MyWeightTopKLoss_Absolutly, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(reduce=False)

        self.MaxClutterNum = MaxClutterNum
        self.ProtectedArea = ProtectedArea
        self.gamma=gamma
        self.alpha=alpha
        self.size_average = size_average
        if isinstance(alpha, (float, int)): self.alpha=torch.Tensor([alpha(0), alpha(1)])
        if isinstance(alpha, list): self.alpha=torch.Tensor(alpha)


    def forward(self, input, target):

        if input.dim() > 4:
            input = torch.squeeze(input, 2)

        ## target surrounding = 2
        template = torch.ones(1, 1, 2*self.ProtectedArea+1, 2*self.ProtectedArea+1).to(input.device)
        target_prot = F.conv2d(target.float(), template, stride=1, padding=self.ProtectedArea)
        target_prot = (target_prot > 0).float()

        with torch.no_grad():
            loss_wise = self.bce_loss(input, target)
            loss_p = loss_wise * (1 - target_prot)
            batch_l = loss_p.shape[0]
            Wgt = torch.zeros(batch_l, 1, 512, 512)
            for ls in range(batch_l):
                loss_ls = loss_p[ls, :, :, :].reshape(-1)
                loss_topk, indices = torch.topk(loss_ls, self.MaxClutterNum)
                for i in range(self.MaxClutterNum):
                    Wgt[ls, 0, indices[i] // 512, indices[i] % 512] = 1

            WgtData_New = Wgt.to(input.device) + target.float()
            WgtData_New[WgtData_New > 1] = 1

        logpt = F.logsigmoid(input)
        logpt_bk = F.logsigmoid(-input)
        pt = logpt.data.exp()
        pt_bk = 1 - logpt_bk.data.exp()
        loss = -self.alpha[1]*(1-pt)**self.gamma*target*logpt - self.alpha[0]*pt_bk**self.gamma*(1-target)*logpt_bk

        loss = loss * WgtData_New

        return loss.sum()

        # if input.dim()>2:
        #     input=input.view(input.size(0), input.size(1),-1)   # N,C,D,H,W=>N,C,D*H*W
        #     input=input.transpose(1,2)                          # N,C,D*H*W=>N, D*H*W, C
        #     input=input.contiguous().view(-1,input.size(2))     # N,D*H*W,C=>N*D*H*W, C
        #
        #     WgtData_New = WgtData_New.view(WgtData_New.size(0), WgtData_New.size(1), -1)    # N,C,D,H,W=>N,C,D*H*W
        #     WgtData_New = WgtData_New.transpose(1, 2)                               # N,C,D*H*W=>N, D*H*W,C
        #     WgtData_New = WgtData_New.contiguous().view(-1, WgtData_New.size(2))        # N,D*H*W,C=>N*D*H*W,C
        #
        # target = target.view(-1,1)           #N,D,H,W=>1,N*D*H*W
        # logpt = F.log_softmax(input)
        # logpt = logpt.gather(1,target)
        # logpt=logpt*WgtData_New             #weight
        # logpt=logpt.view(-1)
        # pt=logpt.data.exp()
        #
        # if self.alpha is not None:
        #     if self.alpha.type()!=input.data.type():
        #         self.alpha=self.alpha.type_as(input.data).to(input.device)
        #     at=self.alpha.gather(0,target.data.view(-1))
        #     logpt=logpt*at
        #
        # loss=-1*(1-pt)**self.gamma*logpt
        #
        # return loss.sum()


