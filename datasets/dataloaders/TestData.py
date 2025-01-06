from PIL import Image, ImageOps, ImageFilter
from torch.utils.data.dataset import Dataset
import random
import numpy as np
import  torch
from utils.transforms import get_img_norm_cfg, Normalized, random_crop, flip, Imageflip, PadImg, resize_mask_to_img
import os


def get_all_images_in_folder(folder_path):
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.lower().endswith(valid_extensions)]
    return image_files

class Test_Dataset(Dataset):

    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, dataset_name, train, base_size, crop_size, img_norm_cfg=None):
        super(Test_Dataset, self).__init__()
        self.train = train
        val_img_ids = get_all_images_in_folder(dataset_dir)
        self._items = val_img_ids
        self.images = dataset_dir
        self.base_size = base_size
        self.crop_size = crop_size

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def _sync_transform(self, img, mask):
        
        base_size = self.base_size
        if img.size != tuple(base_size):
            img  = img.resize (base_size, Image.BILINEAR)
            mask = mask.resize(base_size, Image.NEAREST)
        img, mask = np.array(img), np.array(mask, dtype=np.float32)       
        return img, mask

    def __getitem__(self, idx):

        img_id = self._items[idx]
        img_path = self.images + '/' + img_id

        img = Image.open(img_path).convert('I') 

        imgx= np.array(img, dtype=np.float32)
        
        img = Normalized(imgx, self.img_norm_cfg)
                    
        img_size = img.shape
        img = PadImg(img)

        img = img[np.newaxis, :]
        img = torch.from_numpy(np.ascontiguousarray(img))
        imgx=torch.from_numpy(np.ascontiguousarray(imgx[np.newaxis, :]))

        return (img, imgx), img_size, img_id

    def __len__(self):
        return len(self._items)
