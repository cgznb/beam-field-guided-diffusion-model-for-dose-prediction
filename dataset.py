from torch.utils.data import Dataset
import os
import torch
import numpy as np
from albumentations import (ShiftScaleRotate, Compose, HorizontalFlip, VerticalFlip)
import cv2

class Dataset_PSDM_train(Dataset):
    def __init__(self, data_root='data'):
        self.file_dir_list = []
        self.file_name_list = []
        self.file_dir_list = self.file_dir_list + [data_root] * len(os.listdir(os.path.join(data_root, 'CT')))
        self.file_name_list.extend(os.listdir(os.path.join(data_root, 'CT')))

        self.transforms = Compose([
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=90, p=0.3, value=None,
                             mask_value=None, border_mode=cv2.BORDER_REPLICATE, interpolation=cv2.INTER_NEAREST),
            HorizontalFlip(p=0.3), VerticalFlip(p=0.3)], p=0.8)
        self.len = len(self.file_name_list)

    def __getitem__(self, idx):
        file_dir = self.file_dir_list[idx]
        file_name = self.file_name_list[idx]
        ct = np.load(os.path.join(file_dir, 'CT', file_name))[:, :, np.newaxis]
        dose = np.load(os.path.join(file_dir, 'dose', file_name))[:, :, np.newaxis]
        dose_crt = np.load(os.path.join(file_dir, 'dose_crt', file_name))[:, :, np.newaxis]

        Mask_beam = np.load(os.path.join(file_dir, 'Mask_beam', file_name))[:, :, np.newaxis]
        Mask_body = np.load(os.path.join(file_dir, 'Mask_body', file_name))[:, :, np.newaxis]
        Mask_gtv = np.load(os.path.join(file_dir, 'Mask_gtv', file_name))[:, :, np.newaxis]
        Mask_Kidney_L = np.load(os.path.join(file_dir, 'Mask_Kidney_L', file_name))[:, :, np.newaxis]
        Mask_Kidney_R = np.load(os.path.join(file_dir, 'Mask_Kidney_R', file_name))[:, :, np.newaxis]
        Mask_Liver = np.load(os.path.join(file_dir, 'Mask_Liver', file_name))[:, :, np.newaxis]
        Mask_SpinalCord = np.load(os.path.join(file_dir, 'Mask_SpinalCord', file_name))[:, :, np.newaxis]
        Mask_Stomach = np.load(os.path.join(file_dir, 'Mask_Stomach', file_name))[:, :, np.newaxis]
        Mask_Heart = np.load(os.path.join(file_dir, 'Mask_Heart', file_name))[:, :, np.newaxis]
        Mask_ptv = np.load(os.path.join(file_dir, 'Mask_ptv', file_name))[:, :, np.newaxis]

        PSDM_beam = np.load(os.path.join(file_dir, 'PSDM_beam', file_name))[:, :, np.newaxis]
        PSDM_body = np.load(os.path.join(file_dir, 'PSDM_body', file_name))[:, :, np.newaxis]
        PSDM_gtv = np.load(os.path.join(file_dir, 'PSDM_gtv', file_name))[:, :, np.newaxis]
        PSDM_Kidney_L = np.load(os.path.join(file_dir, 'PSDM_Kidney_L', file_name))[:, :, np.newaxis]
        PSDM_Kidney_R = np.load(os.path.join(file_dir, 'PSDM_Kidney_R', file_name))[:, :, np.newaxis]
        PSDM_Liver = np.load(os.path.join(file_dir, 'PSDM_Liver', file_name))[:, :, np.newaxis]
        PSDM_Stomach = np.load(os.path.join(file_dir, 'PSDM_Stomach', file_name))[:, :, np.newaxis]
        PSDM_ptv = np.load(os.path.join(file_dir, 'PSDM_ptv', file_name))[:, :, np.newaxis]
        PSDM_Heart = np.load(os.path.join(file_dir, 'PSDM_Heart', file_name))[:, :, np.newaxis]
        PSDM_SpinalCord = np.load(os.path.join(file_dir, 'PSDM_SpinalCord', file_name))[:, :, np.newaxis]

        PTVs_mask = Mask_ptv

        data_all = np.concatenate([ct, dose,  PTVs_mask, Mask_body, Mask_gtv, Mask_Kidney_L, Mask_Kidney_R,
                         Mask_Liver, Mask_SpinalCord, Mask_Heart, Mask_Stomach,  PSDM_body, PSDM_gtv,PSDM_Heart,
                                   PSDM_Kidney_L, PSDM_Kidney_R, PSDM_Liver, PSDM_Stomach, PSDM_ptv,  PSDM_SpinalCord], axis=-1)

        data_all = self.transforms(image=data_all)['image']

        ct = torch.from_numpy(data_all[:, :, 0:1]).permute(2, 0, 1)
        dose = torch.from_numpy(data_all[:, :, 1:2]).permute(2, 0, 1)
        dis = torch.from_numpy(data_all[:, :, 2:]).permute(2, 0, 1)
        return ct, dis, dose

    def __len__(self):
        return self.len

class Dataset_PSDM_val(Dataset):
    def __init__(self, data_root='data'):
        self.file_dir_list = []
        self.file_name_list = []
        self.file_dir_list = self.file_dir_list + [data_root] * len(os.listdir(os.path.join(data_root, 'CT')))
        self.file_name_list.extend(os.listdir(os.path.join(data_root, 'CT')))
        self.len = len(self.file_name_list)

    def __getitem__(self, idx):
        file_dir = self.file_dir_list[idx]
        file_name = self.file_name_list[idx]
        ct = np.load(os.path.join(file_dir, 'CT', file_name))[:, :, np.newaxis]
        dose = np.load(os.path.join(file_dir, 'dose', file_name))[:, :, np.newaxis]
        dose_crt = np.load(os.path.join(file_dir, 'dose_crt', file_name))[:, :, np.newaxis]

        Mask_beam = np.load(os.path.join(file_dir, 'Mask_beam', file_name))[:, :, np.newaxis]
        Mask_body = np.load(os.path.join(file_dir, 'Mask_body', file_name))[:, :, np.newaxis]
        Mask_gtv = np.load(os.path.join(file_dir, 'Mask_gtv', file_name))[:, :, np.newaxis]
        Mask_Kidney_L = np.load(os.path.join(file_dir, 'Mask_Kidney_L', file_name))[:, :, np.newaxis]
        Mask_Kidney_R = np.load(os.path.join(file_dir, 'Mask_Kidney_R', file_name))[:, :, np.newaxis]
        Mask_Liver = np.load(os.path.join(file_dir, 'Mask_Liver', file_name))[:, :, np.newaxis]
        Mask_SpinalCord = np.load(os.path.join(file_dir, 'Mask_SpinalCord', file_name))[:, :, np.newaxis]
        Mask_Stomach = np.load(os.path.join(file_dir, 'Mask_Stomach', file_name))[:, :, np.newaxis]
        Mask_ptv = np.load(os.path.join(file_dir, 'Mask_ptv', file_name))[:, :, np.newaxis]
        Mask_Heart = np.load(os.path.join(file_dir, 'Mask_Heart', file_name))[:, :, np.newaxis]

        PSDM_beam = np.load(os.path.join(file_dir, 'PSDM_beam', file_name))[:, :, np.newaxis]
        PSDM_body = np.load(os.path.join(file_dir, 'PSDM_body', file_name))[:, :, np.newaxis]
        PSDM_gtv = np.load(os.path.join(file_dir, 'PSDM_gtv', file_name))[:, :, np.newaxis]
        PSDM_Kidney_L = np.load(os.path.join(file_dir, 'PSDM_Kidney_L', file_name))[:, :, np.newaxis]
        PSDM_Kidney_R = np.load(os.path.join(file_dir, 'PSDM_Kidney_R', file_name))[:, :, np.newaxis]
        PSDM_Liver = np.load(os.path.join(file_dir, 'PSDM_Liver', file_name))[:, :, np.newaxis]
        PSDM_Stomach = np.load(os.path.join(file_dir, 'PSDM_Stomach', file_name))[:, :, np.newaxis]
        PSDM_ptv = np.load(os.path.join(file_dir, 'PSDM_ptv', file_name))[:, :, np.newaxis]
        PSDM_SpinalCord = np.load(os.path.join(file_dir, 'PSDM_SpinalCord', file_name))[:, :, np.newaxis]
        PSDM_Heart = np.load(os.path.join(file_dir, 'PSDM_Heart', file_name))[:, :, np.newaxis]

        PTVs_mask = Mask_ptv

        data_all = np.concatenate(
            [ct, dose,  PTVs_mask,  Mask_body, Mask_gtv, Mask_Kidney_L, Mask_Kidney_R,
             Mask_Liver, Mask_SpinalCord, Mask_Heart, Mask_Stomach,  PSDM_body, PSDM_gtv, PSDM_Heart,
             PSDM_Kidney_L, PSDM_Kidney_R, PSDM_Liver, PSDM_Stomach, PSDM_ptv, PSDM_SpinalCord], axis=-1)

        ct = torch.from_numpy(data_all[:, :, 0:1]).permute(2, 0, 1)
        dose = torch.from_numpy(data_all[:, :, 1:2]).permute(2, 0, 1)
        dis = torch.from_numpy(data_all[:, :, 2:]).permute(2, 0, 1)
        return self.file_name_list[idx], ct, dis, dose

    def __len__(self):
        return self.len