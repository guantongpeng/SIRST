import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


def one_hot_labels(seg_labels, num_classes=4):
    """
    Convert label to one-hot vector.
    """
    one_hot_labels = np.zeros((num_classes, seg_labels.shape[1], seg_labels.shape[2]))
    for i in range(num_classes):
        one_hot_labels[i, :, :] = (seg_labels == i)
    return one_hot_labels

def convert_labels(target):
    """
    将炮仗和碎片标记为目标，其他标记为背景。
    """
    target[target == 2] = 1
    target[target == 3] = 0
    return target


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images), ) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class MWIRSTD_Dataset(Dataset):

    def __init__(self, root: str, base_size=[655, 509], train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.image_root = os.path.join(root, "imgs")
            self.mask_root = os.path.join(root, "masks")
        else:
            self.image_root = os.path.join(root, "Sequences/test", "images")
            self.mask_root = os.path.join(root, "Sequences/test", "masks")
        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".jpg")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".png")]
        assert len(image_names) > 0, f"not find any images in {self.image_root}."

        # check images and mask
        re_mask_names = []
        for p in image_names:
            mask_name = p.replace(".jpg", ".png")
            assert mask_name in mask_names, f"{p} has no corresponding mask."
            re_mask_names.append(mask_name)
        mask_names = re_mask_names

        self.images_path = [os.path.join(self.image_root, n) for n in image_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]
        self.transforms = transforms
        self.base_size = base_size

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        image = Image.open(image_path)
        target = Image.open(mask_path)
        image, target = self._testval_sync_transform(image, target)
        image = np.array(image, dtype=np.float32) / 256
        target = np.array(target)[None, :, :]
        image = image.transpose((2,0,1))
        target = convert_labels(target)
        
        assert target is not None, f"failed to read mask: {mask_path}"

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)
        return batched_imgs, batched_targets

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize(base_size, Image.BILINEAR)
        mask = mask.resize(base_size, Image.NEAREST)

        return img, mask