import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import torchvision.utils as vutils
from torchvision import transforms
import numpy as np
import scipy.io as scio
import time
import os
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import auc
from tensorboardX import SummaryWriter
from datasets import MWIRSTD_Dataset, MIRSDT_Dataset, IRSTD_1K_Dataset, SIRST_Dataset
from models.ISNet.train_ISNet import Get_gradientmask_nopadding, Get_gradient_nopadding
from models.model_config import model_chose, run_model
from utils.losses import loss_chose, AverageMeter
from utils.utils import get_optimizer, seed_pytorch
from utils.metrics import ShootingRules, SigmoidMetric, SamplewiseSigmoidMetric, ROCMetric, PD_FA, mIoU
import json
from pprint import pprint

import warnings
warnings.filterwarnings('ignore')

seed_pytorch(42)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.best_iou = 0
        self.best_nIoU = 0
        self.best_FA = 1e15
        self.best_PD = 0
                
        self.save_folder = args.result_path
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)
            
        # model
        device = torch.device('cuda')
        self.device = device
        
        self.net = model_chose(args.model, args.loss_func, args.deep_supervision)
        if args.device:
            if torch.cuda.device_count() > 1:
                print('use '+str(torch.cuda.device_count())+' gpus')
                self.net = nn.DataParallel(self.net, device_ids=args.device)
                
        self.net = self.net.to(device)
        if args.model == 'SCTransNet':
            from utils.utils import weights_init_kaiming
            self.net.apply(weights_init_kaiming)
            
        train_path = args.datapath + args.dataset + '/'
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.test_path = train_path
        if args.dataset == 'NUDT-MIRSDT':
            self.train_dataset = MIRSDT_Dataset(train_path, train=True, fullSupervision=args.fullySupervised)
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
            self.val_dataset = MIRSDT_Dataset(self.test_path, train=False)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        elif args.dataset == 'MWIRSTD':
            self.train_dataset = MWIRSTD_Dataset(train_path, base_size=args.base_size, train=True)
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
            self.val_dataset = MWIRSTD_Dataset(self.test_path, base_size=args.base_size, train=False)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)           
        elif args.dataset == 'IRSTD-1k':
            self.train_dataset = IRSTD_1K_Dataset(train_path, train=True, base_size=args.base_size, crop_size=args.crop_size)
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
            self.val_dataset = IRSTD_1K_Dataset(self.test_path, train=False)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        elif args.dataset in ['SIRST', 'NUDT-SIRST']:
            spilt_txt = './datasets/split_datasets/' + args.dataset
            self.train_dataset = SIRST_Dataset(train_path, spilt_txt, train=True, transform=input_transform, base_size=args.base_size,crop_size=args.crop_size,)
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
            self.val_dataset = SIRST_Dataset(self.test_path, spilt_txt, train=False, transform=input_transform, base_size=args.base_size,crop_size=args.crop_size,)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        else:
            raise BaseException('Error dataset name, support datasets ["NUDT-MIRSDT", "MWIRSTD", "IRSTD-1k", "SIRST", "NUDT-SIRST"]')
        
         ### Default settings of SCTransNet
        if args.optimizer_name == 'Adam':
            args.optimizer_settings = {'lr': 0.001}
            args.scheduler_name = 'CosineAnnealingLR'
            args.scheduler_settings = {'epochs': args.epochs, 'eta_min': 1e-5, 'last_epoch': -1}

        ### Default settings of DNANet
        if args.optimizer_name == 'Adagrad':
            args.optimizer_settings = {'lr': 0.05}
            args.scheduler_name = 'CosineAnnealingLR'
            args.scheduler_settings = {'epochs': args.epochs, 'min_lr': 1e-5}

        ### Default settings of EGEUNet
        if args.optimizer_name == 'AdamW':
            args.optimizer_settings = {'lr': 0.001, 'betas': (0.9, 0.999), "eps": 1e-8, "weight_decay": 1e-2,
                                    "amsgrad": False}
            args.scheduler_name = 'CosineAnnealingLR'
            args.scheduler_settings = {'epochs': args.epochs, 'T_max': 50, 'eta_min': 1e-5, 'last_epoch': -1}

        self.optimizer, self.scheduler = get_optimizer(self.net, 
                                                       args.optimizer_name, 
                                                       args.scheduler_name, 
                                                       args.optimizer_settings,
                                                       args.scheduler_settings)
        
        # self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)
        # self.scheduler = StepLR(self.optimizer, step_size=args.lr_step, gamma=0.1, last_epoch=-1)

        self.criterion = loss_chose(args)
        self.criterion2 = nn.BCELoss()
        self.eval_metrics = ShootingRules()

        self.loss_list = []
        self.Gain = 100
        self.epoch_loss = 0

        ########### save ############
        self.ModelPath, self.ParameterPath, self.SavePath = generate_savepath(args, 0, 0)
        self.test_save = self.SavePath[0:-1] + '_visualization/'
        self.writeflag = 1
        self.save_flag = 1
        if self.save_flag == 1 and not os.path.exists(self.test_save):
            os.mkdir(self.test_save)

        folder_name = '%s_%s_%s' % (time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())),
                                         args.dataset, args.model)
        self.writer = SummaryWriter(log_dir=self.SavePath)
        self.writer.add_text(folder_name, 'Args:%s, ' % args)

        args_dict = vars(args)
        with open(f'{self.SavePath}/args.json', 'w') as f:
            json.dump(args_dict, f, indent=4)
        pprint(args_dict)
        
    def training(self, epoch):
        args = self.args
        running_loss = 0.0
        self.net.train()
        for i, data in enumerate(tqdm(self.train_loader), 0):
            if i % args.training_rate != 0:
                continue

            SeqData_t, TgtData_t = data
            SeqData, TgtData = Variable(SeqData_t).to(self.device), Variable(TgtData_t).to(self.device)  # b,t,m,n  // b,1,m.n
            self.optimizer.zero_grad()

            outputs = run_model(self.args.dataset, self.net, args.model, SeqData, 0, 0)
            if isinstance(outputs, list):
                if isinstance(outputs[0], tuple):
                    outputs[0] = outputs[0][0]
            elif isinstance(outputs, tuple):
                outputs = outputs[0]

            if 'DNANet' in args.model:
                loss = 0
                if isinstance(outputs, list):
                    for output in outputs:
                        loss += self.criterion(output, TgtData.float())
                    loss /= len(outputs)
                else:
                    loss = self.criterion(outputs, TgtData.float())
            elif 'ISNet' in args.model and args.loss_func == 'fullySup1':   ## and 'ISNet_woTFD' not in args.model
                edge = torch.cat([TgtData, TgtData, TgtData], dim=1).float()  # b, 3, m, n
                gradmask = Get_gradientmask_nopadding()
                edge_gt = gradmask(edge)
                loss_io = self.criterion(outputs[0], TgtData.float())
                if args.fullySupervised:
                    loss_edge = 10 * self.criterion2(torch.sigmoid(outputs[1]), edge_gt) + self.criterion(outputs[1], edge_gt)
                else:
                    loss_edge = 10 * self.criterion2(torch.sigmoid(outputs[1]), edge_gt) + self.criterion(outputs[1], edge_gt.float())
                if 'DTUM' in args.model or not args.fullySupervised:
                    alpha = 0.1
                else:
                    alpha = 1
                loss = loss_io + alpha * loss_edge
            elif 'UIU' in args.model:
                if 'fullySup2' in args.loss_func:
                    loss0, loss = self.criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], TgtData.float())
                    if not args.deep_supervision:
                        loss = loss0   ## without SDS
                else:
                    loss = 0
                    if not args.deep_supervision:
                        loss = self.criterion(outputs[0], TgtData.float())
                    else:
                        for output in outputs:
                            loss += self.criterion(output, TgtData.float())
            else:
                loss = self.criterion(outputs, TgtData.float())

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()

        self.epoch_loss = running_loss / i
        lr_now= trainer.optimizer.param_groups[0]['lr']
        print(f'model: {args.model}, loss: {args.loss_func}, epoch: {epoch + 1}, loss: {self.epoch_loss:.5f}, lr:{lr_now}')
        
        # if trainer.optimizer.param_groups[0]['lr'] > args.lrate_min:
        #     self.scheduler.step()

        self.loss_list.append(self.epoch_loss)

        self.writer.add_scalar('Losses/train loss',self.epoch_loss, epoch)
        self.writer.add_scalar('Learning rate/', trainer.optimizer.param_groups[0]['lr'], epoch)       

    def validation(self, epoch):
        args = self.args
        self.net.eval()
        eval_losses = []
        
        ROC  = ROCMetric(1, 10)
        miou = mIoU(1)
        pd_fa = PD_FA(1, 10, args.crop_size)
        iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        
        tbar = tqdm(self.val_loader)
        for i, data in enumerate(tbar):
            with torch.no_grad():
                img, mask = Variable(data[0]).to(self.device), Variable(data[1]).cpu()
                outputs = run_model(self.args.dataset, self.net, args.model, img)
                if 'ISNet' in args.model: 
                    edge_out = torch.sigmoid(outputs[1]).data.cpu().numpy()
                if isinstance(outputs, list):
                    outputs = outputs[0]
                if isinstance(outputs, tuple):
                    Old_Feat = outputs[1]
                    outputs = outputs[0]  
  
                outputs = torch.squeeze(outputs, 2)
                # outputs = torch.sigmoid(outputs)
                # TestOut = Outputs_Max.data
                TestOut = outputs.data.cpu()

            loss = self.criterion(TestOut, mask.float())
            eval_losses.append(loss.item())
            
            iou_metric.update(TestOut, mask)
            self.nIoU_metric.update(TestOut, mask)
            ROC.update(TestOut, mask)
            miou.update(TestOut, mask)
            # import pdb
            # pdb.set_trace()
            pd_fa.update(TestOut, mask)
            FA, PD = pd_fa.get(len(self.val_loader))
            _, mean_IOU = miou.get()
            _, IoU = iou_metric.get()
            _, nIoU = self.nIoU_metric.get()
            eval_loss = np.mean(eval_losses)
            tbar.set_description('Epoch:%3d, eval loss:%f, IoU:%f, nIoU:%f' %(epoch, eval_loss, IoU, nIoU))

        IOU_part = '_IoU-%.4f_nIoU-%.4f_' % (IoU, nIoU)
        if (IoU > self.best_iou) or (nIoU > self.best_nIoU):
            self.savemodel(epoch, True, IOU_part, eval_loss)
            self.best_iou = max(IoU, self.best_iou)
            self.best_nIoU = max(nIoU, self.best_nIoU)
        if FA[0] < self.best_FA:
            self.best_FA = FA[0]
        if PD[0] > self.best_PD:
            self.best_PD = PD[0]      
                 
        print('mIoU', mean_IOU * 1e2)
        print('nIoU', nIoU * 1e2)
        print('Fa', FA[0] * 1e6)
        print('Pd', PD[0] * 1e2)

        # img_grid_i = vutils.make_grid(data, normalize=True, scale_each=True, nrow=8)
        # self.writer.add_image('input img', img_grid_i, global_step=None)  # j 表示feature map数
        # img_grid_o = vutils.make_grid(TestOut, normalize=True, scale_each=True, nrow=8)
        # self.writer.add_image('output img', img_grid_o, global_step=None)  # j 表示feature map数
        # img_gt = vutils.make_grid(mask, normalize=True, scale_each=True, nrow=8)
        # self.writer.add_image('img gt', img_gt, global_step=None)  # j 表示feature map数

        self.writer.add_scalar('Losses/eval_loss', np.mean(eval_losses), epoch)
        self.writer.add_scalar('Eval/IoU', IoU, epoch)
        self.writer.add_scalar('Eval/nIoU', nIoU, epoch)
        self.writer.add_scalar('Best/IoU', self.best_iou, epoch)
        self.writer.add_scalar('Best/nIoU', self.best_nIoU, epoch)
        self.writer.add_scalar('Best/FA', self.best_FA, epoch)
        self.writer.add_scalar('Best/PD', self.best_PD, epoch)
        # self.writer.add_scalar('FA_PD', PD, FA)
        # self.writer.add_scalar('FP_TP', ture_positive_rate, false_positive_rate)
        # self.writer.add_scalar('Pre_Recall', recall, precision)  
        
                                                   
    def savemodel(self, epoch, val=False, IOU_part= None, eval_loss=None):
        if val:
            self.ModelPath, self.ParameterPath, self.SavePath = generate_savepath(self.args, epoch, eval_loss, IOU_part)
        else:
            self.ModelPath, self.ParameterPath, self.SavePath = generate_savepath(self.args, epoch, self.epoch_loss)
        torch.save(self.net, self.ModelPath)
        torch.save(self.net.state_dict(), self.ParameterPath)
        print('save net OK in %s' % self.ModelPath)


    def saveloss(self):
        CurTime = time.strftime("%Y%m%d%H%M", time.localtime())
        # print(CurTime)

        ###########save lost_list
        LossMatSavePath = self.SavePath + 'loss_list_' + CurTime + '.mat'
        scio.savemat(LossMatSavePath, mdict={'loss_list': self.loss_list})

        ############plot
        x1 = range(self.args.epochs)
        y1 = self.loss_list
        fig = plt.figure()
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        LossJPGSavePath = self.SavePath + 'train_loss_' + CurTime + '.jpg'
        plt.savefig(LossJPGSavePath)
        # plt.show()
        print('finished Show!')

def generate_savepath(args, epoch, epoch_loss, IOU_part=''):

    timestamp = time.time()
    CurTime = time.strftime("%Y%m%d%H%M", time.localtime(timestamp))

    SavePath = args.result_path + args.dataset  + '_' + args.model + '/'
    ModelPath = SavePath + CurTime + '_net_epoch_' + str(epoch) + '_loss_' + f"{epoch_loss:.4f}" + IOU_part + '.pth'
    ParameterPath = SavePath + 'net_para_' + CurTime + '.pth'

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)

    return ModelPath, ParameterPath, SavePath

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Infrared_target_detection_overall')
    parser.add_argument('--datapath',  type=str, default='../datasets/', help='Dataset path [default: ../dataset/]')
    parser.add_argument('--dataset',   type=str, default='NUDT-MIRSDT', help='Dataset name in ["NUDT-MIRSDT", "MWIRSTD", "IRSTD-1k", "SIRST", "NUDT-SIRST"]')
    parser.add_argument('--training_rate', type=int, default=1, help='Rate of samples in training (1/n) [default: 1]')
    parser.add_argument('--result_path',   type=str, default='./results/', help='Save path [defaule: ./results/]')
    parser.add_argument('--train',    type=int, default=0)
    parser.add_argument('--test',     type=int, default=1)
    parser.add_argument('--pth_path', type=str, default='./model_weights/ResUNet_DTUM.pth', help='Trained model path')
    parser.add_argument('--base_size', nargs='*', type=int, default=[256, 256])
    parser.add_argument('--crop_size', type=int, default=256)
    # train
    parser.add_argument('--model',     type=str, default='ResUNet_DTUM', help='ResUNet_DTUM, DNANet_DTUM, ACM, ALCNet, ResUNet, DNANet, ISNet, UIU, SCTransNet, MiM, MSHNet')
    parser.add_argument('--loss_func', type=str, default='fullySup', help='HPM, FocalLoss, OHEM, fullySup, fullySup1(ISNet), fullySup2(UIU)')
    parser.add_argument('--batchsize', type=int,   default=1)
    parser.add_argument('--epochs',    type=int,   default=20)
    parser.add_argument('--optimizer_name', type=str, default='Adam')
    parser.add_argument('--lr',     type=float, default=0.001)
    parser.add_argument('--lrate_min', type=float, default=1e-5)
    parser.add_argument('--lr_step', type=int, default=40)
    parser.add_argument('--deep_supervision',  default=False)

    # GPU
    parser.add_argument('--device', nargs='*', type=int, default=[0,1,2,3], help='use comma for multiple gpus')
    args = parser.parse_args()

    # the parser
    return args


if __name__ == '__main__':
    args = parse_args()
    # StartTime = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    # print(StartTime)

    # GPU
    # torch.cuda.set_device(args.device)
          
    trainer = Trainer(args)
    if args.train == 1:
        for epoch in range(args.epochs):
            trainer.training(epoch)

            if ((epoch + 1) % 30 == 0) or (epoch == args.epochs - 1):
                trainer.savemodel(epoch)
                if args.dataset == "NUDT-MIRSDT":
                    trainer.NUDT_MIRSDT_validation(epoch)
                else:
                    trainer.validation(epoch)
                    
        # trainer.savemodel()
        trainer.saveloss()
        print('finished training!')
    if args.test == 1:
        #####################################################
        trainer.ModelPath = args.pth_path
        trainer.test_save = trainer.SavePath[0:-1] + '_visualization/'
        trainer.net = torch.load(trainer.ModelPath, map_location=torch.device('cuda'), weights_only=False)
        print('load OK!')
        epoch = args.epochs
        #####################################################
        if args.dataset == "NUDT-MIRSDT":
            trainer.NUDT_MIRSDT_validation(epoch) 
        else:
            trainer.validation(epoch)


# TODO 
# 断点继续训练  