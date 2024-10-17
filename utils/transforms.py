import random
from typing import List, Union
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import os

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.flip_prob = prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize(object):
    def __init__(self, size: Union[int, List[int]], resize_mask: bool = True):
        self.size = size  # [h, w]
        self.resize_mask = resize_mask

    def __call__(self, image, target=None):
        image = F.resize(image, self.size)
        if self.resize_mask is True:
            target = F.resize(target, self.size)

        return image, target


class RandomCrop(object):
    def __init__(self, size: int):
        self.size = size

    def pad_if_smaller(self, img, fill=0):
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        min_size = min(img.shape[-2:])
        if min_size < self.size:
            ow, oh = img.size
            padh = self.size - oh if oh < self.size else 0
            padw = self.size - ow if ow < self.size else 0
            img = F.pad(img, [0, 0, padw, padh], fill=fill)
        return img

    def __call__(self, image, target):
        image = self.pad_if_smaller(image)
        target = self.pad_if_smaller(target)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

def Normalized(img, img_norm_cfg):
    return (img - img_norm_cfg['mean']) / img_norm_cfg['std']

def Denormalization(img, img_norm_cfg):
    return img * img_norm_cfg['std'] + img_norm_cfg['mean']

def random_crop(img, mask, patch_size, pos_prob=None):
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                     mode='constant')  # 将不足 256的一边填充至256
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)),
                      mode='constant')  # label 与image 进行相同的变换
        h, w = img.shape

    while 1:
        h_start = random.randint(0, h - patch_size)
        h_end = h_start + patch_size
        w_start = random.randint(0, w - patch_size)
        w_end = w_start + patch_size

        img_patch = img[h_start:h_end, w_start:w_end]
        mask_patch = mask[h_start:h_end, w_start:w_end]

        if pos_prob == None or random.random() > pos_prob:
            break
        elif mask_patch.sum() > 0:
            break

    return img_patch, mask_patch

class flip(object):
    def __call__(self, input, target):
        if random.random() < 0.5:  # 水平反转
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:  # 垂直反转
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:  # 转置反转
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target

def Imageflip(img, mask):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask

def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img

from scipy.ndimage import zoom

def resize_mask_to_img(mask, img_shape):
    """
    Resize the mask to the shape of the image using scipy.ndimage.zoom.
    
    Parameters:
    - mask: 2D numpy array, the mask to be resized.
    - img_shape: tuple or list, the shape of the image (height, width).
    
    Returns:
    - resized_mask: 2D numpy array, the mask resized to the image shape.
    """
    # Calculate the zoom factors
    zoom_factors = (img_shape[0] / mask.shape[0], img_shape[1] / mask.shape[1])
    
    # Resize the mask
    resized_mask = zoom(mask, zoom_factors, order=0)
    
    return resized_mask

def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name == 'SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1K':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    elif dataset_name == 'SIRST2':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'SIRST3':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST-Sea':
        img_norm_cfg = dict(mean=43.62403869628906, std=18.91838264465332)
    elif dataset_name == 'MWIRSTD':
        img_norm_cfg = dict(mean=81.86100769042969, std=37.245872497558594)        
    elif dataset_name == 'IRDST-real':
        img_norm_cfg = {'mean': 101.54053497314453, 'std': 56.49856185913086}
    else:
        with open(dataset_dir + '/' + dataset_name + '/train.txt', 'r') as f:
            train_list = f.read().splitlines()
        with open(dataset_dir + '/' + dataset_name + '/test.txt', 'r') as f:
            test_list = f.read().splitlines()
        img_list = train_list + test_list
        img_dir = dataset_dir + '/' + dataset_name + '/images/'
        mean_list = []
        std_list = []
                    
        for img_pth in img_list:
            img = Image.open(img_dir +"/"+ img_pth).convert('I')
            try:
                img = Image.open((img_dir + img_pth).replace('//', '/') + '.png').convert('I')
            except:
                try:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.jpg').convert('I')
                except:
                    img = Image.open((img_dir + img_pth).replace('//', '/') + '.bmp').convert('I')
            img = np.array(img, dtype=np.float32)
            mean_list.append(img.mean())
            std_list.append(img.std())
        _mean = float(np.array(mean_list).mean())
        _std = float(np.array(std_list).mean())
        print(_mean, _std)
        img_norm_cfg = dict(mean=_mean, std=_std)
    return img_norm_cfg