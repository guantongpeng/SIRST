from PIL import Image, ImageOps, ImageFilter
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch


class SIRST_Dataset(Dataset):

    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, spilt_txt, train, base_size,crop_size,transform=None,suffix='.png'):
        super(SIRST_Dataset, self).__init__()
        self.train = train
        self.transform = transform
        train_img_ids, val_img_ids = load_dataset(spilt_txt)
        self._items = train_img_ids if train else val_img_ids
        self.masks = dataset_dir+'/'+'masks'
        self.images = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix = suffix

    def _sync_transform(self, img, mask):
        # random mirror
        if self.train:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
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
        else:
            base_size = self.base_size
            if img.size != tuple(base_size):
                img  = img.resize (base_size, Image.BILINEAR)
                mask = mask.resize(base_size, Image.NEAREST)

            # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.            
        return img, mask

    def __getitem__(self, idx):

        img_id = self._items[idx]      # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path = self.images+'/'+img_id+self.suffix   # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix

        img = Image.open(img_path).convert('RGB')    # 由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道
        mask = Image.open(label_path)

        # synchronized transform
        img, mask = self._sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32')/ 255.0
        # img = img.transpose((2,0,1)).astype('float32')
        return img, torch.from_numpy(mask) #img_id[-1]

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
