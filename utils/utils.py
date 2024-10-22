# from SCTransNet
import time
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


def seed_pytorch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)  
    
def get_optimizer(net, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings):
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings['lr'])
    if optimizer_name == 'Adamweight':
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings['lr'], weight_decay=1e-3)

    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=optimizer_settings['lr'])
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=optimizer_settings['lr'],
                                    momentum=0.9,
                                    weight_decay=scheduler_settings['weight_decay'])
    # elif optimizer_name == 'AdamW':
    #     optimizer = torch.optim.AdamW(net.parameters(), lr=optimizer_settings['lr'], betas=optimizer_settings['betas'],
    #                                   eps=optimizer_settings['eps'], weight_decay=optimizer_settings['weight_decay'],
    #                                   amsgrad=optimizer_settings['amsgrad'])

    if scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_settings['step'],
                                                         gamma=scheduler_settings['gamma'])
    # elif scheduler_name == 'DNACosineAnnealingLR':
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'],
    #                                                            eta_min=scheduler_settings['eta_min'])
    elif scheduler_name == 'CosineAnnealingLR':
        warmup_epochs = 10
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'] - warmup_epochs,
                                                                      eta_min=scheduler_settings['eta_min'])
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)
    elif scheduler_name == 'CosineAnnealingLRw50':
        warmup_epochs = 50
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'] - warmup_epochs,
                                                                      eta_min=scheduler_settings['eta_min'])
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                           after_scheduler=scheduler_cosine)

    elif scheduler_name == 'CosineAnnealingLRw0':
        # warmup_epochs = 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'], eta_min=scheduler_settings['eta_min'])
        # scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'] - warmup_epochs,
        #                                                               eta_min=1e-5)
        # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
        #                                    after_scheduler=scheduler_cosine)

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['T_max'],
        #                                                        eta_min=scheduler_settings['eta_min'],
        #                                                        last_epoch=scheduler_settings['last_epoch'])
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_settings['epochs'], eta_min=scheduler_settings['eta_min'])

    return optimizer, scheduler


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    在optimizer中会设置一个基础学习率base lr,
    当multiplier>1时,预热机制会在total_epoch内把学习率从base lr逐渐增加到multiplier*base lr,再接着开始正常的scheduler
    当multiplier==1.0时,预热机制会在total_epoch内把学习率从0逐渐增加到base lr,再接着开始正常的scheduler
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler and (not self.finished):
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
            return [base_lr for base_lr in self.after_scheduler.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        print('warmuping...')
        if self.last_epoch <= self.total_epoch:
            warmup_lr=None
            if self.multiplier == 1.0:
                warmup_lr = [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
            else:
                warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics,epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def generate_savepath(args, epoch, epoch_loss, loss, IOU_part=''):

    timestamp = time.time()
    cur_time = time.strftime("%Y%m%d%H%M", time.localtime(timestamp))

    save_path = args.result_path + args.dataset  + '_' + args.model_name + '_'+ str(loss).split('()')[0] + '/models/'
    model_path = save_path + f'{args.batchsize}' + '_' + cur_time + '_net_epoch_' + str(epoch) + '_loss_' + f"{epoch_loss:.4f}" + IOU_part + '.pth'
    parameter_path = save_path + f'{args.batchsize}' + '_' + cur_time + '_net_para_' + str(epoch) + '_loss_' + f"{epoch_loss:.4f}" + IOU_part + '.pth'

    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path, exist_ok=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    return model_path, parameter_path, save_path

def delete_pth_files(folder_path):
    pth_files = glob.glob(os.path.join(folder_path, '*.pth'))
    
    for file_path in pth_files:
        try:
            os.remove(file_path)
            print(f"delete pth file: {file_path}")
        except Exception as e:
            print(f"delete pth file  {file_path} error: {e}")
            
def save_pred_imgs(output, mask, save_path, img_id):
    pred_img = ToPILImage()((output[0, 0, :, :])).convert('1')
    pred_imgs_path = save_path + '/' + 'pred_imgs'
    if not os.path.exists(pred_imgs_path):
        os.makedirs(pred_imgs_path)
    pred_img.save(pred_imgs_path + '/' + img_id[0] + '.png')
    
    img_width, img_height = pred_img.size
    mixed_img = Image.new('1', (img_width * 3 + 2 * 6, img_height + 2 * 2 + 20))
    draw = ImageDraw.Draw(mixed_img)
    border_width = 2
    font = ImageFont.load_default()
    text_height = 20
    label_img = ToPILImage()(mask[0, 0, :, :]).convert('1')
    diff_img = ToPILImage()(np.abs(output - mask)[0, 0, :, :]).convert('1')
    label_text = "Label"
    pred_text = "Prediction"
    diff_text = "Difference"
    mixed_img.paste(label_img, (border_width, text_height + border_width))
    draw.text((border_width, border_width // 2), label_text, font=font, fill=1)
    draw.rectangle([(border_width, border_width), (img_width + border_width, img_height + border_width + text_height)], outline=1)
    mixed_img.paste(pred_img, (img_width + border_width * 5, text_height + border_width))
    draw.text((img_width + border_width * 3, border_width // 2), pred_text, font=font, fill=1)
    draw.rectangle([(img_width + border_width * 3, border_width), (img_width * 2 + border_width * 3, img_height + border_width + text_height)], outline=1)
    mixed_img.paste(diff_img, (img_width * 2 + border_width * 7, text_height + border_width))
    draw.text((img_width * 2 + border_width * 5, border_width // 2), diff_text, font=font, fill=1)
    draw.rectangle([(img_width * 2 + border_width * 5, border_width), (img_width * 3 + border_width * 5, img_height + border_width + text_height)], outline=1)
    mixed_imgs_path = save_path + '/' + 'mix_imgs'
    if not os.path.exists(mixed_imgs_path):
        os.makedirs(mixed_imgs_path)
    mixed_img.save(mixed_imgs_path + '/' + img_id[0] + '_mix.png')