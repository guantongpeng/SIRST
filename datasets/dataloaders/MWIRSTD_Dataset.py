import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
from utils.transforms import get_img_norm_cfg, Normalized, random_crop, flip, Imageflip, PadImg, resize_mask_to_img


def one_hot_labels(seg_labels, num_classes=4):
    """
    Convert label to one-hot vector.
    """
    one_hot_labels = np.zeros((num_classes, seg_labels.shape[1], seg_labels.shape[2]))
    for i in range(num_classes):
        one_hot_labels[i, :, :] = (seg_labels == i)
    return one_hot_labels

def convert_labels(mask):
    """
    将炮仗和碎片标记为目标，其他标记为背景。
    """
    mask[mask == 2] = 1
    mask[mask == 3] = 0
    return mask


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images), ) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class MWIRSTD_Dataset(Dataset):

    def __init__(self, root: str, train, base_size, crop_size, img_norm_cfg=None):
        
        self.base_size = base_size
        self.crop_size = crop_size
        self.train = train
        self.tranform = flip()
        
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.image_root = os.path.join(root, "images")
            self.mask_root = os.path.join(root, "masks")
        else:
            self.image_root = os.path.join(root, "Sequences/test", "images")
            self.mask_root = os.path.join(root, "Sequences/test", "masks")
        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".jpg")]
        assert len(image_names) > 0, f"not find any images in {self.image_root}."
        self.image_names = image_names
        # check images and mask

        self._items = [n for n in image_names]

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg("MWIRSTD", self.image_root)
        else:
            self.img_norm_cfg = img_norm_cfg
        

    def __getitem__(self, idx):
        
        img_id = self._items[idx]
        image_path = os.path.join(self.image_root, img_id)
        mask_path = os.path.join(self.mask_root, img_id.replace(".jpg", ".png"))
        img = Image.open(image_path).convert('I') 
        mask = Image.open(mask_path)
   
        # image, mask = self._testval_sync_transform(image, mask)
        # image = np.array(image, dtype=np.float32) / 256
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
    
        mask = np.array(mask)
        if img.shape != mask.shape:
            # print(f'Mask shape is diff to img, Resize mask shape from {mask.shape} to {img.shape}')
            mask = resize_mask_to_img(mask, img.shape)

        if self.train:
            img, mask = random_crop(img, mask, self.crop_size, pos_prob=0.5)  # 把短的一边先pad至256 把长的一边 随机裁出256  输出 256 256
            img, mask = self.tranform(img, mask)
        else: # 尺寸没有保持（256， 256）
            
            img = img[10:-60,50:-30]
            mask = mask[10:-60,50:-30]
            img_size = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
                        
        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        mask = convert_labels(mask)
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))        

        if self.train:
            return img, mask
        
        return (img, mask), img_size, img_id

    def __len__(self):
        return len(self.image_names)

    @staticmethod
    def collate_fn(batch):
        images, masks = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_masks = cat_list(masks, fill_value=0)
        return batched_imgs, batched_masks

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize(base_size, Image.BILINEAR)
        mask = mask.resize(base_size, Image.NEAREST)

        return img, mask