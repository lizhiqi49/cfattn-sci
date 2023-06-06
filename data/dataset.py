import os
import glob
import imageio
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset

"""
Data directory structure:
- root
    - train
        - category1
            - crop1
                - 0.png
                - 1.png
                ...
                - 7.png
                - measure.npy
            - crop2
            ...
        - category2
        ...
    - val
"""



class RGBSingleFrameDataset(Dataset):

    def __init__(
        self,
        root,
        split
    ):
        super().__init__()

        self.root = root
        self.split = split
        self.gt_paths = glob.glob(
            os.path.join(root, split, '*', '*', '*.png')
        )
        # self.masks = scio.loadmat('./mask.mat')
        self.masks = np.load('./masks.npy')

    def __len__(self):
        return  len(self.gt_paths)
    
    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        data_dir, gt_name = os.path.split(gt_path)
        frame_idx = eval(gt_name.split('.')[0])
        meas = np.load(os.path.join(data_dir, 'measure.npy'))

        # Load, tensorize and normalize
        gt = imageio.imread(gt_path)
        gt = torch.from_numpy(gt).float()
        gt = (gt - 127.5) / 127.5

        mask = torch.from_numpy(self.masks[frame_idx]).float()
        meas = torch.from_numpy(meas).permute(2, 0, 1)  # (c h w)
        cond = {'mask': mask, 'measure': meas}

        return gt, cond

        