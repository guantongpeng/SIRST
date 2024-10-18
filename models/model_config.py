import torch
from models.DNANet import DNANet
from models.ResUNet import ResUNet
from models.ACM import ASKCResUNet as ACM
from models.ALCNet import ASKCResNetFPN as ALCNet
from models.DNANet import Res_CBAM_block
from models.ResUNet import Res_block
from models.UIUNet.uiunet import UIUNET
from models.SCTransNet import SCTransNet
from models.RDIAN import RDIAN
from models.UNet import U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet
from models.EGEUNet import EGEUNet
from models.EffisegNet import EffiSegNet
# from models.ISNet.ISNet import ISNet
from models.MSHNet import MSHNet
from models.MiM import MiM
from mmcv.utils import Config
# from models.RevCol.mmsegmentation.tools import revcol_seg
from models.RevCol2 import revcol_seg


def model_chose(model_name, deep_supervision=True, num_classes=1, input_channels=1):

    if model_name == 'ACM':
        net = ACM(in_channels=input_channels,
                layers=[3] * 3,
                fuse_mode='AsymBi',
                tiny=False,
                classes=num_classes)

    elif model_name == 'ALCNet':
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

    elif model_name == 'DNANet':
        net = DNANet(num_classes=num_classes,
                    input_channels=input_channels,
                    block=Res_CBAM_block,
                    num_blocks=[2, 2, 2, 2],
                    nb_filter=[16, 32, 64, 128, 256],
                    deep_supervision=deep_supervision)

    elif model_name == 'ResUNet':
        net = ResUNet(num_classes=num_classes,
                    input_channels=input_channels,
                    block=Res_block,
                    num_blocks=[2, 2, 2, 2],
                    nb_filter=[8, 16, 32, 64, 128])
        

    # elif model == 'ISNet':
    #     net = ISNet(layer_blocks=[4] * 3,
    #                 channels=[8, 16, 32, 64],
    #                 num_classes=num_classes)

    elif model_name == 'UIUNet':
        net = UIUNET(input_channels=input_channels, 
                    out_ch=num_classes, 
                    deep_supervision=deep_supervision)
        
    elif model_name == 'SCTransNet':
        net = SCTransNet(mode='train', deep_supervision=deep_supervision)       

    elif model_name == 'EGEUNet':
        net = EGEUNet()  
        
    elif model_name == 'UNet':
        net = U_Net(n1=32)

    elif model_name == 'AttUNet':
        net = AttU_Net(n1=32)

    elif model_name == 'R2UNet':
        net = R2U_Net(n1=32)
        
    elif model_name == 'R2AttUNet':
        net = R2AttU_Net(n1=32)
        
    elif model_name == 'NestedUNet':
        net = NestedUNet(n1=32)

    elif model_name == 'EffiSegNet':
        net = EffiSegNet(deep_supervision=deep_supervision)
                
    elif model_name == 'RDIAN':
        net = RDIAN()    
                    
    elif model_name == 'MSHNet':
        net = MSHNet(input_channels=input_channels)

    elif model_name == 'MiM':
        net = MiM([2]*3,[8, 16, 32, 64, 128])     
    
    elif model_name == 'RevCol':
        cfg = Config.fromfile('/home/guantp/Infrared/SIRST/models/RevCol/mmsegmentation/configs/revcol/test.py')
        net = revcol_seg.get_model(cfg)
    elif model_name == 'RevCol2':
        channels = [64, 128, 256, 512]
        layers = [2, 2, 4, 2]
        num_subnet = 4
        drop_path=0.1
        net = revcol_seg.FullNet(channels, layers, num_subnet, num_classes=num_classes, drop_path = drop_path)   

    else:
        raise UnboundLocalError(f"Error model name {model_name}!!!")

    return net

def run_model(net, model_name, input):

    if model_name == 'ISNet':
        outputs = net(input)
        
    elif model_name == 'RevCol2':

        outputs = net(input)[1]
        outputs = torch.sigmoid(outputs[-1])
        # outputs = [torch.sigmoid(output) for output in outputs[::-1]]
        
    else:
        outputs = net(input)

        if isinstance(outputs, list) or isinstance(outputs, tuple):
            outputs = [torch.sigmoid(output) for output in outputs]
        else:
            outputs = torch.sigmoid(outputs)
    return outputs
