import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) 
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out
        
        
class Conv_d11(nn.Module):
    def __init__(self):            
        super(Conv_d11, self).__init__()
        kernel = [[-1, 0, 0, 0, 0],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [0,0,0,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2) 
        
class Conv_d12(nn.Module):
    def __init__(self):            
        super(Conv_d12, self).__init__()
        kernel = [[0, 0, -1, 0, 0],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [0,0,0,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)    


class Conv_d13(nn.Module):
    def __init__(self):            
        super(Conv_d13, self).__init__()
        kernel = [[0, 0, 0, 0, -1],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [0,0,0,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)    


class Conv_d14(nn.Module):
    def __init__(self):            
        super(Conv_d14, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,-1],
                  [0, 0, 0,0,0],
                  [0,0,0,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)    


class Conv_d15(nn.Module):
    def __init__(self):            
        super(Conv_d15, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [0,0,0,0,-1]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)         
        
class Conv_d16(nn.Module):
    def __init__(self):            
        super(Conv_d16, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [0,0,-1,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)   

class Conv_d17(nn.Module):
    def __init__(self):            
        super(Conv_d17, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0,0,0],
                  [0, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [-1,0,0,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)         
        
class Conv_d18(nn.Module):
    def __init__(self):            
        super(Conv_d18, self).__init__()
        kernel = [[0, 0, 0, 0, 0],
                  [0, 0, 0,0,0],
                  [-1, 0, 1,0,0],
                  [0, 0, 0,0,0],
                  [0,0,0,0,0]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, input):
        return F.conv2d(input, self.weight, padding=2)         
    

class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)

def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())

class NewBlock(nn.Module):
    def __init__(self, in_channels, stride,kernel_size,padding):
        super(NewBlock, self).__init__()
        reduced_channels = int(in_channels/2)
        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.layer2 = conv_batch(reduced_channels, in_channels, kernel_size=kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out        

class RDIAN(nn.Module):
    def __init__(self):
    
        super(RDIAN, self).__init__()        
        self.conv1 = conv_batch(1, 16)
        self.conv2 = conv_batch(16, 32, stride=2)       
        self.residual_block0 = self.make_layer(NewBlock, in_channels=32, num_blocks=1, kernel_size=1,padding=0,stride=1)
        self.residual_block1 = self.make_layer(NewBlock, in_channels=32, num_blocks=2, kernel_size=3,padding=1,stride=1)
        self.residual_block2 = self.make_layer(NewBlock, in_channels=32, num_blocks=2, kernel_size=5,padding=2,stride=1)
        self.residual_block3 = self.make_layer(NewBlock, in_channels=32, num_blocks=2, kernel_size=7,padding=3,stride=1)
        self.cbam  = CBAM(32, 32)        
        self.conv_cat = conv_batch(4*32, 32, 3, padding=1)
        self.conv_res = conv_batch(16, 32, 1, padding=0)
        self.relu = nn.ReLU(True)
        
        self.d11=Conv_d11()
        self.d12=Conv_d12()
        self.d13=Conv_d13()
        self.d14=Conv_d14()
        self.d15=Conv_d15()
        self.d16=Conv_d16()
        self.d17=Conv_d17()
        self.d18=Conv_d18()

        self.head = _FCNHead(32, 1)

    def forward(self, x):
        _, _, hei, wid = x.shape
        d11 = self.d11(x)
        d12 = self.d12(x)
        d13 = self.d13(x)
        d14 = self.d14(x)
        d15 = self.d15(x)
        d16 = self.d16(x)
        d17 = self.d17(x)
        d18 = self.d18(x)
        md = d11.mul(d15) + d12.mul(d16) + d13.mul(d17) + d14.mul(d18)
        md = F.sigmoid(md)
        
        out1= self.conv1(x)        
        out2 = out1.mul(md)       
        out = self.conv2(out1 + out2)
            
        c0 = self.residual_block0(out)
        c1 = self.residual_block1(out)
        c2 = self.residual_block2(out)
        c3 = self.residual_block3(out)
 
        x_cat = self.conv_cat(torch.cat((c0, c1, c2, c3), dim=1)) #[16,32,240,240]
        x_a = self.cbam(x_cat)
        
        temp = F.interpolate(x_a, size=[hei, wid], mode='bilinear')
        temp2 = self.conv_res(out1)
        x_new = self.relu( temp + temp2)
        self.x_new = x_new
        pred = self.head(x_new)

        return pred
             
    def make_layer(self, block, in_channels, num_blocks, stride, kernel_size, padding):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, stride, kernel_size, padding))
        return nn.Sequential(*layers)
