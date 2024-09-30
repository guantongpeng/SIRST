import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import scipy.io as scio


class MIRSDT_Dataset(Dataset):
    
    def __init__(self, root: str, train: bool = True, fullSupervision=False):
        txt_path = "train.txt" if train else "test.txt"
        txtpath = root + txt_path
        txt = np.loadtxt(txtpath, dtype=bytes).astype(str)
        self.imgs_arr = txt
        self.root = root
        self.fullSupervision = fullSupervision
        self._mean = 105.4025
        self._std = 26.6452

    def __getitem__(self, index):
        img_path_mix = self.root + self.imgs_arr[index]

        # Read Mix
        MixData_mat = scio.loadmat(img_path_mix)
        # try except: MixData_mat = scio.loadmat(img_path_mix, verify_compressed_data_integrity=False)

        MixData_Img = MixData_mat.get("Mix")  # MatData
        MixData_Img = MixData_Img.astype(np.float32)

        # Read Label/Tgt
        img_path_tgt = img_path_mix.replace(".mat", ".png")
        # print(img_path_mix)
        if self.fullSupervision:
            img_path_tgt = img_path_tgt.replace("Mix", "masks")
            LabelData_Img = Image.open(img_path_tgt)
            LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0
        else:
            img_path_tgt = img_path_tgt.replace("Mix", "masks_centroid")
            LabelData_Img = Image.open(img_path_tgt)
            LabelData_Img = np.array(LabelData_Img, dtype=np.float32) / 255.0

        # Mix preprocess
        MixData_Img = (MixData_Img - self._mean) / self._std
        MixData = torch.from_numpy(MixData_Img)

        MixData_out = torch.unsqueeze(MixData[-5:, :, :], 0)  # the last five frame

        [m_L, n_L] = np.shape(LabelData_Img)
        if m_L == 512 and n_L == 512:
            # Tgt preprocess
            LabelData = torch.from_numpy(LabelData_Img)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            return MixData_out, TgtData_out

        else:
            # Tgt preprocess
            [n, t, m_M, n_M] = np.shape(MixData_out)
            LabelData_Img_1 = np.zeros([512, 512])
            LabelData_Img_1[0:m_L, 0:n_L] = LabelData_Img
            LabelData = torch.from_numpy(LabelData_Img_1)
            TgtData_out = torch.unsqueeze(LabelData, 0)
            MixData_out_1 = torch.zeros([n, t, 512, 512])
            MixData_out_1[0:n, 0:t, 0:m_M, 0:n_M] = MixData_out
            return MixData_out_1, TgtData_out
        
    def __len__(self):
        return len(self.imgs_arr)
    

