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
from datasets import MWIRSTD_Dataset, MIRSDT_Dataset, IRSTD_1K_Dataset, SIRST_Dataset, Test_Dataset
from models.model_config import model_chose, run_model
from utils.losses import loss_chose, DPLoss
from utils.utils import get_optimizer, seed_pytorch, generate_savepath, delete_pth_files, save_pred_imgs, weights_init_kaiming, save_test_pred_imgs
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
            
        # dataloader
        train_path = args.datapath + args.dataset + '/'
        self.test_path = train_path

        if args.dataset in ['SIRST', 'NUDT-SIRST', 'IRSTD-1k']:
            spilt_txt = './datasets/split_datasets/' + args.dataset
            self.val_dataset = SIRST_Dataset(self.test_path, spilt_txt, dataset_name=args.dataset, train=False, base_size=args.base_size,crop_size=args.crop_size,test=True)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        elif args.dataset == 'TestData':
            self.val_dataset = Test_Dataset(args.datapath, dataset_name=args.dataset, train=False, base_size=args.base_size,crop_size=args.crop_size,)
            self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        else:
            raise BaseException(f'Error dataset name {args.dataset}, support datasets ["NUDT-MIRSDT", "MWIRSTD", "IRSTD-1k", "SIRST", "NUDT-SIRST"]')

        # model
        self.net = model_chose(args.model_name, args.deep_supervision)
        self.net.apply(weights_init_kaiming)
        self.device = torch.device('cuda')
        print('use '+str(torch.cuda.device_count())+' gpus')     
        self.net = nn.DataParallel(self.net)          
        self.net = self.net.to(self.device)    

        ########### save ############
        self.model_path, self.parameter_path, self.save_path = generate_savepath(args, 0, 0, 'Test')
        args_dict = vars(args)
        with open(f'{self.save_path}/args.json', 'w') as f:
            json.dump(args_dict, f, indent=4)
            
        # print("="*50 + "  config  " + "="*50)
        # pprint(args_dict)
        # print("="*50 + "  config  " + "="*50)
        

        self.test_log_file = open(self.save_path + 'test_log.txt', 'w')
        self.test_log_file.write(f"{args.pth_path}\n")


    def validation(self):
        args = self.args
        self.net.eval()
        
        tbar = tqdm(self.val_loader)
        for i, (data, img_ori, img_id) in enumerate(tbar):
            with torch.no_grad():
                img, mask = Variable(data[0]).to(self.device), Variable(data[1]).cpu()
                outputs = run_model(self.net, args.model_name, img)
                     
                if isinstance(outputs, list) or isinstance(outputs, tuple):
                    outputs = outputs[0]
                outputs = torch.squeeze(outputs, 2)
                output = outputs.data.cpu()  

            if args.save_pred_img:
                save_test_pred_imgs(img_ori, (output > args.threshold).to(torch.float), mask.to(torch.float), self.save_path, img_id)
                # save_pred_imgs(output, mask, self.save_path, img_id)

                                                   

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Infrared_target_detection_overall')
    parser.add_argument('--datapath',  type=str, default='../datasets/', help='Dataset path [default: ../dataset/]')
    parser.add_argument('--dataset',   type=str, default='SIRST', help='Dataset name in ["MWIRSTD", "IRSTD-1k", "SIRST", "NUDT-SIRST"]')
    parser.add_argument('--result_path',   type=str, default='./results/', help='Save path [defaule: ./results/]')

    parser.add_argument('--pth_path', type=str, default='./model_weights/ResUNet_DTUM.pth', help='Trained model path')
    parser.add_argument('--base_size', nargs='*', type=int, default=[256, 256])
    parser.add_argument('--crop_size', type=int, default=256)
    
    parser.add_argument('--mask', type=bool, default=True)
    # train
    parser.add_argument('--model_name',     type=str, default='SCTransNet', help='ACM, ALCNet, ResUNet, DNANet, ISNet, UIUNet, SCTransNet, MiM, MSHNet...')
    parser.add_argument('--batchsize', type=int,   default=1)
    parser.add_argument('--deep_supervision',  default=True)
    # test
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
    parser.add_argument('--save_pred_img',  default=False)
    args = parser.parse_args()

    # the parser
    return args


if __name__ == '__main__':
    args = parse_args()         
    trainer = Trainer(args)
    
    
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
    trainer.validation()

    trainer.test_log_file.close()

