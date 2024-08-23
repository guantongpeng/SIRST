import os
from PIL import Image
import cv2
import torch.utils.data as data
import numpy as np

def one_hot_labels(seg_labels, num_classes=4):
    """
    Convert label to one-hot vector.
    :param label: input label. [8, 1, 655, 509]
    :param num_classes: number of classes.
    :return: one-hot vector. [8, 4, 655, 509]
    """
    one_hot_labels = np.zeros((seg_labels.shape[0], seg_labels.shape[1], num_classes))
    for i in range(4):
        one_hot_labels[:, :, i] = (seg_labels == i).astype(int)
    return one_hot_labels

class MWIRSTDDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.image_root = os.path.join(root, "data", "imgs")
            self.mask_root = os.path.join(root, "data", "masks")
        else:
            self.image_root = os.path.join(root, "data/MWIRSTD/Sequences/test", "images")
            self.mask_root = os.path.join(root, "data/MWIRSTD/Sequences/test", "masks")
        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        image_names = [p for p in os.listdir(self.image_root) if p.endswith(".jpg")]
        mask_names = [p for p in os.listdir(self.mask_root) if p.endswith(".png")]
        assert len(image_names) > 0, f"not find any images in {self.image_root}."

        # check images and mask
        re_mask_names = []
        for p in image_names:
            mask_name = p.replace(".jpg", ".png")
            # import pdb; pdb.set_trace()
            assert mask_name in mask_names, f"{p} has no corresponding mask."
            re_mask_names.append(mask_name)
        mask_names = re_mask_names

        self.images_path = [os.path.join(self.image_root, n) for n in image_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in mask_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]
        image = Image.open(image_path)
        assert image is not None, f"failed to read image: {image_path}"
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        # h, w, _ = image.shape
        image = np.array(image, dtype=np.float32) / 256
        # print(image.shape)
        # print(image.max())

        target = Image.open(mask_path)
        target = np.array(target)
        
        target = one_hot_labels(target, num_classes=4)
        # print(target.max())
        # print(target.shape)
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


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
    train_dataset = MWIRSTDDataset("./", train=True)
    print(len(train_dataset))

    val_dataset = MWIRSTDDataset("./", train=False)
    print(len(val_dataset))

    i, t = train_dataset[0]
