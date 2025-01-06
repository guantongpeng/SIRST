import torch
import argparse
from collections import OrderedDict
import sys
sys.path.append('../') 
from models.model_config import model_chose


def load_model(net, model_path):
    print('loading model!')
    try:
        state_dict = torch.load(model_path)
        try: 
            net.load_state_dict(state_dict)
        except:
            new_state_dict = OrderedDict()
            for k, v in state_dict['state_dict'].items():
                name = "module." + k
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)   
    except:
        net = torch.load(model_path, map_location=torch.device('cuda'), weights_only=False)
    return net

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Infrared_target_detection_overall')
    parser.add_argument('--model_name', type=str, default='DNANet', help='ACM, ALCNet, ResUNet, DNANet, ISNet, UIU, SCTransNet, MiM, MSHNet')
    parser.add_argument('--pth_path', type=str, default='./model_weights/ResUNet_DTUM.pth', help='Trained model path')
    parser.add_argument('--img_size', nargs='*', type=int, default=[256, 256])
    parser.add_argument('--opset_version', type=int, default=14)
    args = parser.parse_args()

    # the parser
    return args

if __name__ == '__main__':
    
    args = parse_args()

    w, h = args.img_size
    x = torch.randn(1, 1, w, h).cuda()
    
    net = model_chose(args.model_name, deep_supervision=False, h=h, w=w)
    net = load_model(net, args.pth_path)
    net.eval()
    with torch.no_grad():
        out = net(x)
        
    model_path = args.pth_path.replace(".pth", ".onnx")
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    
    with torch.no_grad():
        torch.onnx.export(
        net,
        x,
        model_path,
        opset_version= args.opset_version,
        input_names=['input'],
        output_names=['output'])
        
    print('export onnx OK!')

# TODO 字典加载模式导出有bug