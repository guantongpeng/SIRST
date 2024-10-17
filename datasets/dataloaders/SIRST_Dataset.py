from PIL import Image, ImageOps, ImageFilter
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from utils.transforms import get_img_norm_cfg, Normalized, random_crop, flip, Imageflip, PadImg, resize_mask_to_img

class SIRST_Dataset(Dataset):

    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, spilt_txt, dataset_name, train, base_size, crop_size, suffix='.png', img_norm_cfg=None):
        super(SIRST_Dataset, self).__init__()
        self.train = train
        train_img_ids, val_img_ids = load_dataset(spilt_txt)
        self._items = train_img_ids if train else val_img_ids
        self.masks = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = flip()
        
    def _sync_transform(self, img, mask):
        # random mirror
        if self.train:
            img, mask = Imageflip(img, mask)       
            crop_size = self.crop_size
            # random scale (short edge)
            w, h = img.size
            if h > w:
                long_size = random.randint(int(self.base_size[1] * 0.5), int(self.base_size[1] * 2.0))
                oh = long_size
                ow = int(1.0 * w * long_size / h + 0.5)
                short_size = ow
            else:
                long_size = random.randint(int(self.base_size[0] * 0.5), int(self.base_size[0] * 2.0))
                ow = long_size
                oh = int(1.0 * h * long_size / w + 0.5)
                short_size = oh
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
            # pad crop
            if short_size < crop_size:
                padh = crop_size - oh if oh < crop_size else 0
                padw = crop_size - ow if ow < crop_size else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            # random crop crop_size
            w, h = img.size
            x1 = random.randint(0, w - crop_size)
            y1 = random.randint(0, h - crop_size)
            img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            # gaussian blur as in PSP
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(
                    radius=random.random()))
            img, mask = np.array(img), np.array(mask, dtype=np.float32)

            if random.random() < 0.5:
                img = img.transpose((1, 0, 2))
                mask = np.array(mask).transpose((1, 0))
        else:
            base_size = self.base_size
            if img.size != tuple(base_size):
                img  = img.resize (base_size, Image.BILINEAR)
                mask = mask.resize(base_size, Image.NEAREST)
            img, mask = np.array(img), np.array(mask, dtype=np.float32)       
        return img, mask

    def __getitem__(self, idx):

        img_id = self._items[idx]
        img_path = self.images + '/' + img_id + self.suffix
        label_path = self.masks + '/' + img_id + self.suffix

        img = Image.open(img_path).convert('I') 
        mask = Image.open(label_path)

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        # synchronized transform
        # img = Image.fromarray(img, mode='RGB') # 会格式化到0-255，导致出现问题
        # img, mask = self._sync_transform(img, mask)  # 弃用
        # mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0
        # img = img.transpose((2,0,1)).astype('float32')

        if img.shape != mask.shape:
            # print(f'Mask shape is diff to img, Resize mask shape from {mask.shape} to {img.shape}')
            mask = resize_mask_to_img(mask, img.shape)
                     
        if self.train:
            img, mask = random_crop(img, mask, self.crop_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256
            img, mask = self.tranform(img, mask)
        else: # 尺寸没有保持（256， 256）
            img_size = img.shape # 这一行放在后面会降低FA
            img = PadImg(img)
            mask = PadImg(mask)

        img, mask = img[np.newaxis, :], mask[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))

        if self.train:
            return img, mask
        
        return (img, mask), img_size, img_id

    def __len__(self):
        return len(self._items)

def load_dataset(spilt_txt):
    train_txt = spilt_txt + '/train.txt'
    test_txt  = spilt_txt + '/test.txt'
    train_img_ids = []
    val_img_ids = []
    with open(train_txt, "r") as f:
        line = f.readline()
        while line:
            train_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return train_img_ids, val_img_ids

# SIRST 


# 源代码测试结果
# mIoU: 78.8243 | nIoU: 85.2112 | Pd: 97.6303| Fa: 12.5943 |F1: 88.1579

# 本代码原始测试结果
# mIoU 78.98748741550409
# nIoU 83.84858732292348
# F1_score 88.25983177387666
# Fa 12.575925051510989
# Pd 97.0

# 使用了padimg，修改了两张尺寸不对图的mask resize to img
# mIoU 82.32352100600171
# nIoU 86.15752541433453
# F1_score 90.30436896081267
# Fa 8.216271033653845
# Pd 98.5781990521327