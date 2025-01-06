import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# class GlobalFilter(nn.Module):
#     def __init__(self, dim, filiter_num):
#         super().__init__()

#         self.filiter_num = filiter_num
#         self.merge = nn.Conv2d(dim * self.filiter_num, dim, 1, bias=False)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x_fft = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')

#         if self.filiter_num == 1:
#             weight = torch.view_as_complex(nn.Parameter(torch.randn(C, H, W // 2 + 1, 2, dtype=torch.float32) * 0.02)).cuda() 
#             x_filtered = x_fft * weight
#             x = torch.fft.irfft2(x_filtered, s=(H, W), dim=(2, 3), norm='ortho')
            
#         else:
#             weight = torch.view_as_complex(nn.Parameter(torch.randn(self.filiter_num, C, H, W // 2 + 1, 2, dtype=torch.float32) * 0.02)).cuda() 
#             outputs = torch.stack([torch.fft.irfft2(x_fft * weight[i], s=(H, W), dim=(2, 3), norm='ortho')
#                                     for i in range(self.filiter_num)], dim=1)
#             x = self.merge(outputs.view(outputs.shape[0], -1, H, W))

#         return x

# class GlobalFilter(nn.Module):
#     def __init__(self, dim, filiter_num):
#         super().__init__()
#         # self.H, self.W  = size, size
#         self.filiter_num = filiter_num
#         if self.filiter_num == 1:
#             self.weight = torch.view_as_complex(nn.Parameter(torch.randn(self.filiter_num, dim, self.H, self.W // 2 + 1, 2, dtype=torch.float32) * 0.02))
#         else:
#             self.weight = torch.view_as_complex(nn.Parameter(torch.randn(dim, self.H, self.W // 2 + 1, 2, dtype=torch.float32) * 0.02))
#         self.merge = nn.Conv2d(dim * self.filiter_num, dim, 1, bias=False)

#     def forward(self, x):
        
#         x_fft = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')

#         if self.filiter_num == 1:
            
#             x_filtered = x_fft * self.weight
#             x = torch.fft.irfft2(x_filtered, s=(self.H, self.W), dim=(2, 3), norm='ortho')
            
#         else:
            
#             outputs = torch.stack([torch.fft.irfft2(x_fft * self.weight[i], s=(self.H, self.W), dim=(2, 3), norm='ortho')
#                                     for i in range(self.filiter_num)], dim=1)
#             x = self.merge(outputs.view(outputs.shape[0], -1, self.H, self.W))

#         return x

# class GlobalFilter(nn.Module):
#     def __init__(self, dim, filiter_num):
#         super().__init__()
#         self.filiter_num = filiter_num
#         mid_channel = dim * self.filiter_num
#         self.conv = nn.Conv2d(2 * dim, mid_channel, kernel_size = 1)
#         self.bn = nn.BatchNorm2d(mid_channel)
#         self.relu = nn.ReLU(inplace = True)
#         self.merge = nn.Conv2d(mid_channel, dim * 2, kernel_size = 1)

#     def forward(self, x):  
#         import pdb
#         # pdb.set_trace()
#         batch, _, h, w = x.size()
        
#         x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        
#         x_fft_real = torch.unsqueeze(torch.real(x), dim=-1)
#         x_fft_imag = torch.unsqueeze(torch.imag(x), dim=-1)
#         x = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        
#         x = x.permute(0, 1, 4, 2, 3).contiguous()
#         x = x.view((batch, -1,) + x.size()[3:])
        
#         x = self.conv(x)
#         x = self.relu(self.bn(x))
#         x = self.merge(x)
        
#         x = x.view((batch, -1, 2,) + x.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()    
#         x = torch.view_as_complex(x)
        
#         x = torch.fft.irfft2(x, s=(h, w), dim=(2, 3), norm='ortho')
#         return x

# class GlobalFilter(nn.Module):
#     def __init__(self, dim, h=14, w=8):
#         super().__init__()
#         self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
#         self.w = w
#         self.h = h

#     def forward(self, x, spatial_size=None):
#         B, N, C = x.shape
#         if spatial_size is None:
#             a = b = int(math.sqrt(N))
#         else:
#             a, b = spatial_size

#         x = x.view(B, a, b, C)

#         x = x.to(torch.float32)

#         x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
#         weight = torch.view_as_complex(self.complex_weight)
#         x = x * weight
#         x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

#         x = x.reshape(B, N, C)

#         return x
    
# class GlobalFilter(nn.Module):
#     def __init__(self, dim, filiter_num):
#         super().__init__()
#         self.filiter_num = filiter_num
#         mid_channel = 2 * dim * self.filiter_num
#         self.conv1 = nn.Conv2d(2 * dim, mid_channel, kernel_size = 1)
#         self.bn1 = nn.BatchNorm2d(mid_channel)
#         self.relu1 = nn.ReLU(inplace = True)
#         self.conv2 = nn.Conv2d(mid_channel, 2 * dim, kernel_size = 1)
#         self.bn2 = nn.BatchNorm2d(2 * dim)
#         self.relu2 = nn.Tanh()
#         # self.merge = nn.Conv2d(mid_channel // 2, dim, kernel_size = 1)

#     def forward(self, x):  
#         import pdb
#         # pdb.set_trace()
#         batch, c, h, w = x.size()
        
#         x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        
#         x_real_imag = torch.concat((torch.real(x), torch.imag(x)), dim=1)
#         weight = self.relu1(self.bn1(self.conv1(x_real_imag)))
#         weight = self.relu2(self.bn2(self.conv2(weight)))
#         # weight = weight.view((batch, self.filiter_num, 2*c, h, w//2+1)).permute(1, 0, 2, 3, 4)
#         weight = weight.view((batch, 2, c, h, w//2+1)).permute(0, 2, 3, 4, 1).contiguous()
        
#         weight = torch.view_as_complex(weight)
#         x = x * weight
#         x = torch.fft.irfft2(x, s=(h, w), dim=(2, 3), norm='ortho')
#         return x

        # 只修改幅度不修改相位、同时修改幅度和相位
        # 使用不同的网络分别处理幅度和相位
        # 使用生成参数（自适应大小）和随机参数

# class GlobalFilter(nn.Module):
#     def __init__(self, dim, filiter_num):
#         super().__init__()
#         self.filiter_num = filiter_num
#         mid_channel = 2 * dim * self.filiter_num
#         self.conv1 = nn.Conv2d(2 * dim, mid_channel, kernel_size = 1)
#         self.bn1 = nn.BatchNorm2d(mid_channel)
#         self.relu1 = nn.ReLU(inplace = True)
#         self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size = 1)
#         # self.bn2 = nn.BatchNorm2d(mid_channel)
#         # self.relu2 = nn.Tanh()
#         self.merge = nn.Conv2d(mid_channel // 2, dim, kernel_size = 1)

#     def forward(self, x):  
#         import pdb
#         # pdb.set_trace()
#         batch, c, h, w = x.size()
        
#         x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        
#         x_real_imag = torch.concat((torch.real(x), torch.imag(x)), dim=1)
#         weight = self.relu1(self.bn1(self.conv1(x_real_imag)))
#         weight = self.conv2(weight)
#         weight = weight.view((batch, self.filiter_num, 2, c, h, w//2+1)).permute(1, 0, 3, 4, 5, 2).contiguous()
#         # weight = weight.view((batch, 2, c, h, w//2+1)).permute(0, 2, 3, 4, 1).contiguous()
        
#         weight = torch.view_as_complex(weight)

#         x = torch.stack([torch.fft.irfft2(x * weight[i], s=(h, w), dim=(-2, -1), norm='ortho')
#                                     for i in range(self.filiter_num)], dim=1).view(batch, self.filiter_num*c, h, w)
#         x = self.merge(x)
#         return x


# class GlobalFilter(nn.Module):
#     def __init__(self, dim, filiter_num):
#         super().__init__()
#         self.filiter_num = filiter_num
#         mid_channel = 2 * dim * self.filiter_num
#         self.conv1 = nn.Conv2d(2 * dim, mid_channel, kernel_size = 1)
#         self.bn1 = nn.BatchNorm2d(mid_channel)
#         self.relu1 = nn.ReLU(inplace = True)
#         self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size = 1)
#         # self.bn2 = nn.BatchNorm2d(mid_channel)
#         # self.relu2 = nn.Tanh()
#         self.merge = nn.Conv2d(mid_channel // 2, dim, kernel_size = 1)

#     def forward(self, x):  
#         import pdb
#         # pdb.set_trace()
#         batch, c, h, w = x.size()
        
#         x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        
#         x_real_imag = torch.concat((torch.real(x), torch.imag(x)), dim=1)
#         weight = self.relu1(self.bn1(self.conv1(x_real_imag)))
#         weight = self.conv2(weight)
#         weight = weight.view((batch, self.filiter_num, 2, c, h, w//2+1)).permute(1, 0, 3, 4, 5, 2).contiguous()
#         # weight = weight.view((batch, 2, c, h, w//2+1)).permute(0, 2, 3, 4, 1).contiguous()
        
#         weight = torch.view_as_complex(weight)

#         x = torch.stack([torch.fft.irfft2(x * weight[i], s=(h, w), dim=(-2, -1), norm='ortho')
#                                     for i in range(self.filiter_num)], dim=1).view(batch, self.filiter_num*c, h, w)
#         x = self.merge(x)
#         return x

# class GlobalFilter(nn.Module):
#     def __init__(self, dim, filiter_num=4):
#         super().__init__()
#         self.filiter_num = filiter_num
#         mid_channel = 2 * dim * self.filiter_num
#         self.bn1 = nn.BatchNorm2d(2 * dim)
#         self.conv1 = nn.Conv2d(2 * dim, mid_channel, kernel_size = 1)
#         self.relu1 = nn.ReLU(inplace = True)
#         self.conv2 = nn.Conv2d(mid_channel, mid_channel, kernel_size = 1)
#         self.bn2 = nn.BatchNorm2d(mid_channel)
#         self.relu2 = nn.ReLU(inplace = True)
#         self.merge = nn.Conv2d(mid_channel // 2, dim, kernel_size = 1)

#     def forward(self, x):  
#         import pdb
#         # pdb.set_trace()
#         batch, c, h, w = x.size()
        
#         x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        
#         x_real_imag = torch.concat((torch.real(x), torch.imag(x)), dim=1)
#         weight = self.relu1(self.conv1(self.bn1(x_real_imag)))
#         weight = self.relu2(self.bn2(self.conv2(weight)))
#         import pdb
#         pdb.set_trace()
#         weight = weight.view((batch, self.filiter_num, 2, c, h, w//2+1)).permute(1, 0, 3, 4, 5, 2).contiguous()
#         # weight = weight.view((batch, 2, c, h, w//2+1)).permute(0, 2, 3, 4, 1).contiguous()
        
#         weight = torch.view_as_complex(weight)

#         x = torch.stack([torch.fft.irfft2(x * weight[i], s=(h, w), dim=(-2, -1), norm='ortho')
#                                     for i in range(self.filiter_num)], dim=1).view(batch, self.filiter_num*c, h, w)
#         x = self.merge(x)
#         return x

# class GlobalFilter(nn.Module):
#     def __init__(self, dim, filiter_num=4):
#         super().__init__()
#         self.filiter_num = filiter_num        
#         self.conv1 = nn.Conv2d(2 * dim, 2 * dim, kernel_size = 1)
#         self.bn1 = nn.BatchNorm2d(2 * dim)
#         self.relu1 = nn.ReLU(inplace = True)
        

#     def forward(self, x):  

#         batch, c, h, w = x.size()
#         x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
    
#         x_real_imag = torch.concat((torch.real(x), torch.imag(x)), dim=1)
#         weight = self.relu1(self.bn1(self.conv1(x_real_imag)))

#         weight = weight.view((batch, 2, c, h, w//2+1)).permute(0, 2, 3, 4, 1).contiguous()
#         # weight = weight.view((batch, 2, c, h, w//2+1)).permute(0, 2, 3, 4, 1).contiguous()
        
#         weight = torch.view_as_complex(weight)

#         x = torch.fft.irfft2(x * weight, s=(h, w), dim=(-2, -1), norm='ortho')
#         return x


# class GlobalFilter(nn.Module):
#     def __init__(self, dim, filiter_num=4):
#         super().__init__()
#         self.filiter_num = filiter_num        
#         self.conv_real = nn.Conv2d(dim, dim, kernel_size = 1)
#         self.bn_real = nn.BatchNorm2d(dim)
#         self.relu_real = nn.ReLU(inplace = True)

#         self.conv_imag = nn.Conv2d(dim, dim, kernel_size = 1)
#         self.bn_imag = nn.BatchNorm2d(dim)
#         self.relu_imag = nn.ReLU(inplace = True)

#     def forward(self, x):  

#         batch, c, h, w = x.size()
#         x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        
#         weight_imag = self.relu_imag(self.bn_imag(self.conv_imag(torch.randn_like(torch.imag(x)))))
#         weight_real = self.relu_real(self.bn_real(self.conv_real(torch.randn_like(torch.real(x)))))

#         weight = torch.complex(weight_real, weight_imag)

#         x = torch.fft.irfft2(x * weight, s=(h, w), dim=(-2, -1), norm='ortho')
#         return x


# class GlobalFilter(nn.Module):
#     def __init__(self, dim, h=14, w=8):
#         super().__init__()
#         self.complex_weight = nn.Parameter(torch.view_as_complex(torch.randn(1, dim, h, w//2 + 1, 2, dtype=torch.float32) * 0.02), requires_grad=True)
#         self.w = w
#         self.h = h

#     def forward(self, x):
#         # B, C, H, W = x.shape
#         x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
#         x = x * self.complex_weight
#         # import pdb
#         # pdb.set_trace()
#         x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(2, 3), norm='ortho')
#         return x

# class GlobalFilter(nn.Module):
#     def __init__(self, dim, h=14, w=8):
#         super().__init__()
#         self.conv = nn.Conv2d(dim, 2 * dim, kernel_size = 1)
#         self.bn = nn.BatchNorm2d(dim * 2)
#         self.w = w
#         self.h = h

#     def forward(self, x):
#         B, C, H, W = x.shape
#         weight = self.bn(self.conv(x))
#         x = torch.fft.fft2(x, dim=(2, 3), norm='ortho')

#         complex_weight = torch.view_as_complex(weight.view(B, 2, C, H, W).permute(0,2,3,4,1).contiguous())
#         x = x * complex_weight
        
#         x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(2, 3), norm='ortho')
#         return x

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.conv1 = nn.Conv2d(2 * dim, 2 * dim, kernel_size = 1)
        self.bn1 = nn.BatchNorm2d(dim * 2)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(2*dim, 2 * dim, kernel_size = 1)
        self.bn2 = nn.BatchNorm2d(dim * 2)
        # self.act2 = nn.tanh()
        
        self.w = w
        self.h = h

    def forward(self, x):
        B, C, H, W = x.shape
        
        x = torch.fft.fft2(x, dim=(2, 3), norm='ortho')
        x = torch.concat((torch.real(x), torch.imag(x)), dim=1)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = torch.view_as_complex(x.view(B, 2, C, H, W).permute(0,2,3,4,1).contiguous())
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        return x

# class GlobalFilter(nn.Module):
#     def __init__(self, dim, h=14, w=8):
#         super().__init__()
#         self.conv1 = nn.Conv2d(2 * dim, 2 * dim, kernel_size = 1)
#         self.bn1 = nn.BatchNorm2d(dim * 2)
#         self.act1 = nn.ReLU()

#         # self.conv2 = nn.Conv2d(2*dim, 2 * dim, kernel_size = 1)
#         # self.bn2 = nn.BatchNorm2d(dim * 2)
#         # self.act2 = nn.ReLU()
        
#         self.w = w
#         self.h = h

#     def forward(self, x):
#         B, C, H, W = x.shape
        
#         x = torch.fft.fft2(x, dim=(2, 3), norm='ortho')
#         x = torch.concat((torch.real(x), torch.imag(x)), dim=1)
#         x = self.act1(self.bn1(self.conv1(x)))
#         # x = self.bn2(self.conv2(x))
#         x = torch.view_as_complex(x.view(B, 2, C, H, W).permute(0,2,3,4,1).contiguous())
#         x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
#         return x

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, map_size=(256, 256), FFT_num = 4, stride = 1):
        super(ResNet, self).__init__()
        
        self.globalfilter = GlobalFilter(in_channels, map_size[0], map_size[1])
        # import pdb
        # pdb.set_trace()
        # for name, param in self.globalfilter.named_parameters():
        #     print(name, param.shape)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
        self.conv_fft_merge = nn.Sequential(
                nn.Conv2d(out_channels + in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x

        global_out = self.globalfilter(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out

        out = torch.cat([out, global_out], dim=1)
        out = self.conv_fft_merge(out)
        
        if self.shortcut is not None:
            residual = self.shortcut(x)
            

        out += residual
        out = self.relu(out)
        return out

class TFMSHNet(nn.Module):
    def __init__(self, input_channels, block=ResNet, deep_supervision=True, h=512, w=512):
        super().__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]

        map_sizes = [(h, w), (h//2, w//2), (h//4, w//4), (h//8, w//8), (h//16, w//16)]
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_16 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block, 1, map_sizes[0])
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0], map_sizes[1])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1], map_sizes[2])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2], map_sizes[3])
     
        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3], map_sizes[4])
        
        self.decoder_3 = self._make_layer(param_channels[3]+param_channels[4], param_channels[3], block, param_blocks[2], map_sizes[3])
        self.decoder_2 = self._make_layer(param_channels[2]+param_channels[3], param_channels[2], block, param_blocks[1], map_sizes[2])
        self.decoder_1 = self._make_layer(param_channels[1]+param_channels[2], param_channels[1], block, param_blocks[0], map_sizes[1])
        self.decoder_0 = self._make_layer(param_channels[0]+param_channels[1], param_channels[0], block, 1, map_sizes[0])

        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final = nn.Conv2d(4, 1, 3, 1, 1)
        self.deep_supervision = deep_supervision


    def _make_layer(self, in_channels, out_channels, block, block_num=1, map_size=(256, 256)):
        layer = []        
        layer.append(block(in_channels, out_channels, map_size))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels, map_size))
        return nn.Sequential(*layer)

    def forward(self, x):

        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))

        
        if self.deep_supervision:
            mask0 = self.output_0(x_d0)
            mask1 = self.output_1(x_d1)
            mask2 = self.output_2(x_d2)
            mask3 = self.output_3(x_d3)
            output = self.final(torch.cat([mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)], dim=1))
            return output
            # return mask0, self.up(mask1), self.up_4(mask2), self.up_8(mask3)
    
        else:
            output = self.output_0(x_d0)
            return output

       
    