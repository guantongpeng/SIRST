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
from sklearn.metrics import auc
from tensorboardX import SummaryWriter
from datasets import MWIRSTD_Dataset, MIRSDT_Dataset, IRSTD_1K_Dataset, SIRST_Dataset
from models.model_config import model_chose, run_model
from utils.losses import loss_chose, DPLoss
from utils.utils import get_optimizer, seed_pytorch, generate_savepath, delete_pth_files, save_pred_imgs, weights_init_kaiming
from utils.metrics import SigmoidMetric, SamplewiseSigmoidMetric, ROCMetric, PD_FA, mIOU, ROCMetric05
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
        self.net = model_chose(args.model_name, args.deep_supervision)
        self.device = torch.device('cuda')
        print('use '+str(torch.cuda.device_count())+' gpus')
        self.net.apply(weights_init_kaiming)
        self.net = nn.DataParallel(self.net)          
        self.net = self.net.to(self.device)

        # loss
        if args.model_name == 'ISNet':
            self._criterion = loss_chose('ISNetLoss')
        elif args.model_name == 'SCTransNet':
            self._criterion = loss_chose('BCELoss')
        else:
            self._criterion = loss_chose(args.loss_func)
        self.criterion = DPLoss(self._criterion, )
        self.loss_list = []
        self.epoch_loss = 0
                
        # dataloader
        train_path = args.datapath + args.dataset + '/'
        self.test_path = train_path
        if args.dataset == 'NUDT-MIRSDT':
            self.train_dataset = MIRSDT_Dataset(train_path, train=True, fullSupervision=args.fullySupervised)
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
            self.val_dataset = MIRSDT_Dataset(self.test_path, train=False)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        elif args.dataset == 'MWIRSTD':
            self.train_dataset = MWIRSTD_Dataset(train_path, train=True, base_size=args.base_size,crop_size=args.crop_size)
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
            self.val_dataset = MWIRSTD_Dataset(self.test_path,  train=False, base_size=args.base_size,crop_size=args.crop_size)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)           
        elif args.dataset == 'IRSTD-1k':
            self.train_dataset = IRSTD_1K_Dataset(train_path, train=True, base_size=args.base_size, crop_size=args.crop_size)
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
            self.val_dataset = IRSTD_1K_Dataset(self.test_path, train=False)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        elif args.dataset in ['SIRST', 'NUDT-SIRST']:
            spilt_txt = './datasets/split_datasets/' + args.dataset
            self.train_dataset = SIRST_Dataset(train_path, spilt_txt, dataset_name=args.dataset, train=True, base_size=args.base_size,crop_size=args.crop_size,)
            # self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
            self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchsize, shuffle=True)
            self.val_dataset = SIRST_Dataset(self.test_path, spilt_txt, dataset_name=args.dataset, train=False, base_size=args.base_size,crop_size=args.crop_size,)
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
            args.scheduler_settings = {'epochs': args.epochs, 'eta_min': 1e-5}

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

        ########### save ############
        self.model_path, self.parameter_path, self.save_path = generate_savepath(args, 0, 0)
        args_dict = vars(args)
        with open(f'{self.save_path}/args.json', 'w') as f:
            json.dump(args_dict, f, indent=4)
            
        # print("="*50 + "  config  " + "="*50)
        # pprint(args_dict)
        # print("="*50 + "  config  " + "="*50)
        
        if args.test and (not args.train):
            self.test_log_file = open(self.save_path + 'test_log.txt', 'w')
            self.test_log_file.write(f"{args.pth_path}\n")
        else:
            self.writer = SummaryWriter(log_dir=self.save_path)
            folder_name = '%s_%s_%s' % (time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())),
                                         args.dataset, args.model_name)
            self.writer.add_text(folder_name, 'Args:%s, ' % args)
            self.log_file = open(self.save_path + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            self.log_file.write(f"{args}\n")
        
    def training(self, epoch):
        args = self.args
        total_loss_epoch = []
        self.net.train()
        for i, data in enumerate(tqdm(self.train_loader)):
            input, target = Variable(data[0]).to(self.device), Variable(data[1]).to(self.device)  # b,t,m,n  // b,1,m.n
            outputs = run_model(self.net, args.model_name, input)

            loss = self.criterion(outputs, target.float())
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss_epoch.append(loss.detach().cpu())
    
        self.epoch_loss = float(np.array(total_loss_epoch).mean())
        lr_now= self.optimizer.param_groups[0]['lr']

        print(f'model: {args.model_name}, loss: {args.loss_func}, epoch: {epoch}, loss: {self.epoch_loss:.5f}, lr:{lr_now}')
        self.log_file.write(f'model: {args.model_name}, loss: {args.loss_func}, epoch: {epoch}, loss: {self.epoch_loss:.5f}, lr:{lr_now}\n')
        
        self.scheduler.step()
        
        self.loss_list.append(self.epoch_loss)

        self.writer.add_scalar('Losses/train loss',self.epoch_loss, epoch)
        self.writer.add_scalar('Learning rate/', self.optimizer.param_groups[0]['lr'], epoch)     

    def validation(self, epoch, test=False):
        args = self.args
        self.net.eval()
        eval_losses = []
        
        ROC  = ROCMetric05(nclass=1, bins=10)
        mIoU_metric = mIOU()
        pd_fa = PD_FA()
        nIoU_metric = SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5)
        
        tbar = tqdm(self.val_loader)
        for i, (data, img_size, img_id) in enumerate(tbar):
            with torch.no_grad():
                img, mask = Variable(data[0]).to(self.device), Variable(data[1]).cpu()
                outputs = run_model(self.net, args.model_name, img)

                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    outputs = outputs[0]
                outputs = torch.squeeze(outputs, 2)
                output = outputs.data.cpu()

            loss = self.criterion(output, mask.float())
            eval_losses.append(loss.item())
            nIoU_metric.update(output, mask)
            ROC.update(output > args.threshold, mask)
            mIoU_metric.update((output > args.threshold), mask)
            pd_fa.update((output[0, 0, :, :] > args.threshold), mask[0, 0, :, :], img_size)
            temp = nIoU_metric.get()
            tbar.set_description(f'{temp}')
            if args.save_pred_img:
                save_pred_imgs((output > args.threshold).to(torch.float), mask.to(torch.float), self.save_path, img_id)
                # save_pred_imgs(output, mask, self.save_path, img_id)
                                    
        PD, FA = pd_fa.get()
        _, mIoU = mIoU_metric.get()
        nIoU = nIoU_metric.get()
        F1_score = ROC.get()[-1]
        eval_loss = np.mean(eval_losses)

        IOU_part = f'_IoU-{mIoU:.4f}_nIoU-{nIoU:.4f}'
        is_best_iou = mIoU > self.best_iou
        is_best_niou = nIoU > self.best_nIoU
        if not test:
            # if  is_best_iou and is_best_niou:
            #     delete_pth_files(args.result_path + args.dataset  + '_' + args.model_name + '/')
            if  is_best_iou or is_best_niou:
                self.savemodel(epoch, True, IOU_part, eval_loss)  
                self.best_iou = max(mIoU, self.best_iou)
                self.best_nIoU = max(nIoU, self.best_nIoU)
                
            self.writer.add_scalar('Losses/eval_loss', np.mean(eval_losses), epoch)
            self.writer.add_scalar('Eval/IoU', mIoU, epoch)
            self.writer.add_scalar('Eval/nIoU', nIoU, epoch)
            self.writer.add_scalar('Best/IoU', self.best_iou, epoch)
            self.writer.add_scalar('Best/nIoU', self.best_nIoU, epoch)
            self.writer.add_scalar('Best/FA', self.best_FA, epoch)
            self.writer.add_scalar('Best/PD', self.best_PD, epoch)
                    
        if FA < self.best_FA:
            self.best_FA = FA
        if PD > self.best_PD:
            self.best_PD = PD     

        _mIoU, _nIoU, _F1_score, _FA, _PD = mIoU * 1e2, nIoU * 1e2, F1_score * 1e2, FA * 1e6, PD * 1e2
        msg = f'Epoch:{epoch}, eval loss:{eval_loss:.4f}, mIoU:{_mIoU:.4f}, nIoU:{_nIoU:.4f},  F1_score:{_F1_score:.4f}, FA:{_FA:.4f}, PD:{_PD:.4f}\n'
        tbar.set_description(msg)
        
        print('loss', eval_loss)                
        print('mIoU', _mIoU)
        print('nIoU', _nIoU)
        print('F1_score', _F1_score)
        print('Fa', _FA)
        print('Pd', _PD)

        try:
            self.test_log_file.write(msg)
        except:
            self.log_file.write(msg)

                                                   
    def savemodel(self, epoch, val=False, IOU_part= None, eval_loss=None):
        if val:
            self.model_path, self.parameter_path, self.save_path = generate_savepath(self.args, epoch, eval_loss, IOU_part)
        else:
            self.model_path, self.parameter_path, self.save_path = generate_savepath(self.args, epoch, self.epoch_loss)
        torch.save(self.net, self.model_path)
        # torch.save(self.net.state_dict(), self.parameter_path)
        print('save net OK in %s' % self.model_path)

    def saveloss(self):
        cur_time = time.strftime("%Y%m%d%H%M", time.localtime())
        # print(cur_time)

        # save lost_list
        LossTxtSave_path = self.save_path + 'loss_list_' + cur_time + '.txt'
        with open(LossTxtSave_path, 'w') as f:
            for loss in self.loss_list:
                f.write(f"{loss}\n")

        # plot
        x1 = range(self.args.epochs)
        y1 = self.loss_list
        plt.figure()
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        loss_save_path = self.save_path + 'train_loss_' + cur_time + '.jpg'
        plt.savefig(loss_save_path)
        # plt.show()
        print('finished Show!')


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Infrared_target_detection_overall')
    parser.add_argument('--datapath',  type=str, default='../datasets/', help='Dataset path [default: ../dataset/]')
    parser.add_argument('--dataset',   type=str, default='SIRST', help='Dataset name in ["MWIRSTD", "IRSTD-1k", "SIRST", "NUDT-SIRST"]')
    parser.add_argument('--result_path',   type=str, default='./results/', help='Save path [defaule: ./results/]')
    parser.add_argument('--train',    type=int, default=0)
    parser.add_argument('--test',     type=int, default=1)
    parser.add_argument('--pth_path', type=str, default='./model_weights/ResUNet_DTUM.pth', help='Trained model path')
    parser.add_argument('--base_size', nargs='*', type=int, default=[256, 256])
    parser.add_argument('--crop_size', type=int, default=256)
    # train
    parser.add_argument('--model_name',     type=str, default='SCTransNet', help='ACM, ALCNet, ResUNet, DNANet, ISNet, UIUNet, SCTransNet, MiM, MSHNet...')
    parser.add_argument('--loss_func', type=str, default='SoftIoULoss', help='HPM, FocalLoss, OHEM, fullySup, fullySup1(ISNet), fullySup2(UIU)')
    parser.add_argument('--batchsize', type=int,   default=1)
    parser.add_argument('--epochs',    type=int,   default=1000)
    parser.add_argument('--optimizer_name', type=str, default='Adam')
    parser.add_argument('--lr',     type=float, default=0.001)
    parser.add_argument('--lrate_min', type=float, default=1e-5)
    parser.add_argument('--lr_step', type=int, default=40)
    parser.add_argument('--deep_supervision',  default=True)
    # test
    parser.add_argument('--test_epoch', type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
    parser.add_argument('--save_pred_img',  default=False)
    args = parser.parse_args()

    # the parser
    return args


if __name__ == '__main__':
    args = parse_args()         
    trainer = Trainer(args)
    
    if args.train == 1:
        for _epoch in range(args.epochs):
            epoch = _epoch + 1 
            trainer.training(epoch)

            if ((epoch) % args.test_epoch == 0) or (epoch == args.epochs):
                
                if args.dataset == "NUDT-MIRSDT":
                    trainer.NUDT_MIRSDT_validation(epoch)
                else:
                    trainer.validation(epoch)       
        
        trainer.savemodel(epoch=args.epochs)
        trainer.saveloss()
        print('finished training!')
    
    if args.test == 1:
        trainer.model_path = args.pth_path
        # trainer.test_save = trainer.save_path[0:-1] + '_visualization/'
        print('loading model!')
        try:
        ######################## SCTransNet Test use origin code #############################
            from collections import OrderedDict
            state_dict = torch.load(trainer.model_path)
            try: 
                trainer.net.load_state_dict(state_dict)
            except:
                new_state_dict = OrderedDict()
                for k, v in state_dict['state_dict'].items():
                    name = "module." + k
                    new_state_dict[name] = v
                trainer.net.load_state_dict(new_state_dict)
        ######################## SCTransNet Test use origin code #############################            
        except:
            trainer.net = torch.load(trainer.model_path, map_location=torch.device('cuda'), weights_only=False)

        print('load model OK!')
        epoch = args.epochs
        trainer.validation(epoch, test=True)
    try:
        trainer.log_file.close()
    except:
        trainer.test_log_file.close()


# TODO 
# 断点续训  
