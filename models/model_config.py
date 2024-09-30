import torch
from torch import nn

from models.DNANet import DNANet, DNANet_DTUM
# from models.ISNet.ISNet import ISNet_DTUM
from models.ResUNet import res_UNet, res_UNet_DTUM
from models.ACM import ACM
from models.ALCNet import ASKCResNetFPN as ALCNet
from models.ALCNet import ALCNet_DTUM
from models.DNANet import Res_CBAM_block
from models.ResUNet import Res_block
# from models.ISNet.ISNet import ISNet, ISNet_woTFD, ISNet_DTUM
# from models.ISNet.train_ISNet import Get_gradient_nopadding
from models.UIUNet.uiunet import UIUNET, UIUNET_DTUM
from models.SCTransNet import SCTransNet
from models.MSHNet import MSHNet
from models.MiM import MiM
from mmcv.utils import Config
# from models.RevCol.mmsegmentation.tools import revcol_seg
# from thop import profile
# from thop import clever_format



def model_chose(model, loss_func, deep_supervision):
    num_classes = 1

    if model == 'ACM':
        net = ACM(in_channels=3,
                  layers=[3] * 3,
                  fuse_mode='AsymBi',
                  tiny=False,
                  classes=num_classes)

    elif model == 'ALCNet':
        net = ALCNet(layers=[4] * 4,
                     channels=[8, 16, 32, 64, 128],
                     shift=13,
                     pyramid_mod='AsymBi',
                     scale_mode='Single',
                     act_dilation=16,
                     fuse_mode='AsymBi',
                     pyramid_fuse='Single',
                     r=2,
                     classes=num_classes)
        
    elif model == 'ALCNet_DTUM':
        net = ALCNet_DTUM(layers=[4] * 4,
                          channels=[8, 16, 32, 64, 128],
                          shift=13,
                          pyramid_mod='AsymBi',
                          scale_mode='Single',
                          act_dilation=16,
                          fuse_mode='AsymBi',
                          pyramid_fuse='Single',
                          r=2,
                          classes=num_classes)

    elif model == 'DNANet':
        net = DNANet(num_classes=num_classes,
                     input_channels=3,
                     block=Res_CBAM_block,
                     num_blocks=[2, 2, 2, 2],
                     nb_filter=[16, 32, 64, 128, 256],
                     deep_supervision=deep_supervision)
        
    elif model == 'DNANet_DTUM':
        net = DNANet_DTUM(num_classes=num_classes,
                          input_channels=3,
                          block=Res_CBAM_block,
                          num_blocks=[2, 2, 2, 2],
                          nb_filter=[16, 32, 64, 128, 256],
                          deep_supervision=deep_supervision)

    elif model == 'ResUNet':
        net = res_UNet(num_classes=num_classes,
                       input_channels=3,
                       block=Res_block,
                       num_blocks=[2, 2, 2, 2],
                       nb_filter=[8, 16, 32, 64, 128])
        
    elif model == 'ResUNet_DTUM':
        net = res_UNet_DTUM(num_classes=num_classes,
                            input_channels=3,
                            block=Res_block,
                            num_blocks=[2, 2, 2, 2],
                            nb_filter=[8, 16, 32, 64, 128])

    # elif model == 'ISNet':
    #     net = ISNet(layer_blocks=[4] * 3,
    #                 channels=[8, 16, 32, 64],
    #                 num_classes=num_classes)
    # # elif model == 'ISNet_woTFD':
    # #     net = ISNet_woTFD(layer_blocks=[4]*3, channels=[8,16,32,64], num_classes=num_classes)
    # elif model == 'ISNet_DTUM':
    #     net = ISNet_DTUM(layer_blocks=[4] * 3,
    #                      channels=[8, 16, 32, 64],
    #                      num_classes=num_classes)

    elif model == 'UIU':
        net = UIUNET(in_ch=3, 
                     out_ch=num_classes, 
                     deep_supervision=deep_supervision)
        
    elif model == 'UIU_DTUM':
        net = UIUNET_DTUM(in_ch=3,
                          out_ch=num_classes,
                          deep_supervision=deep_supervision)
        
    elif model == 'SCTransNet':
        net = SCTransNet(mode='train')       

    elif model == 'MSHNet':
        net = MSHNet(input_channels=3)

    elif model == 'MiM':
        net = MiM([2]*3,[8, 16, 32, 64, 128])     
       
    elif model == 'RevCol':
        cfg = Config.fromfile('/home/guantp/Infrared/SIRST/models/RevCol/mmsegmentation/configs/revcol/test.py')
        net = revcol_seg.get_model(cfg)
        
    else:
        print("Error model name !!!")

    return net


def run_model(dataset, net, model, SeqData, *args):

    # Old_Feat = SeqData[:,:,:-1, :,:] * 0  # interface for iteration input
    # OldFlag = 1  # 1: i

    if model in ['DNANet', 'ResUNet', 'ACM', 'ALCNet', 'SCTransNet', 'RevCol', 'MiM']:  ## or model=='ISNet_woTFD'
        input = SeqData
        if dataset == "NUDT-MIRSDT":
            input = SeqData[:, :, -1, :, :].repeat(1, 3, 1, 1)
        outputs = net(input)
    elif model in ['DNANet_DTUM', 'ResUNet_DTUM', 'ALCNet_DTUM']:
        input = SeqData.repeat(1, 3, 1, 1, 1)
        outputs = net(input, args[0], args[1])

    # elif model == 'ISNet':
    #     input = SeqData[:, :, -1, :, :].repeat(1, 3, 1, 1)
    #     grad = Get_gradient_nopadding()
    #     edge_in = grad(input)
    #     outputs, edge_outs = net(input, edge_in)
    #     outputs = [outputs, edge_outs]
    # elif model == 'ISNet_DTUM':
    #     input = SeqData.repeat(1, 3, 1, 1, 1)
    #     outputs, edge_outs = net(input, args[0], args[1])
    #     outputs = [outputs, edge_outs]

    elif model == 'UIU':
        input = SeqData
        if dataset == "NUDT-MIRSDT":
            input = SeqData[:, :, -1, :, :].repeat(1, 3, 1, 1)
        d0, d1, d2, d3, d4, d5, d6 = net(input)
        outputs = [d0, d1, d2, d3, d4, d5, d6]
    elif model == 'UIU_DTUM':
        input = SeqData.repeat(1, 3, 1, 1, 1)
        d0, d1, d2, d3, d4, d5, d6 = net(input, args[0], args[1])
        outputs = [d0, d1, d2, d3, d4, d5, d6]
        
    # if  args[1] == 0:
    #     if 'DTUM' in model:
    #         flops, params = profile(net, inputs=(input, args[0], args[1]))   # runtimeerror cpu : net.module
    #     elif model == 'ISNet':
    #         flops, params = profile(net, inputs=(input, edge_in))
    #     else:
    #         flops, params = profile(net, inputs=(input, ))
    #     flops, params = clever_format([flops, params], '%.3f')
    #     print(flops, params)

    return outputs
