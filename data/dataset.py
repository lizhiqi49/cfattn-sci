import os
import cv2
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



class DiffusionRGBSingleFrameDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        image_processor,
        down_factor: int = 4
    ):
        super().__init__()

        self.root = root
        self.split = split
        self.gt_paths = glob.glob(
            os.path.join(root, split, '*', '*', '*.png')
        )
        # self.masks = scio.loadmat('./mask.mat')
        self.masks = np.load('data/masks.npy')
        mask_s = self.masks.sum(axis=0)
        index = np.where(mask_s == 0)
        mask_s[index] = 1
        self.mask_s = mask_s.astype(np.float32)

        self.image_processor = image_processor
        self.down_factor = down_factor

    def __len__(self):
        return  len(self.gt_paths)
    
    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        data_dir, gt_name = os.path.split(gt_path)
        frame_idx = eval(gt_name.split('.')[0])
        meas = np.load(os.path.join(data_dir, 'measure.npy'))
        
        # Normalize measure
        meas = meas / self.mask_s[..., None]
        meas = meas / 255.0
        meas_pixel_values = self.image_processor(meas).pixel_values[0]

        # Load, tensorize and normalize
        gt = imageio.imread(gt_path)
        # Downsample
        h, w, c = gt.shape
        gt = cv2.resize(gt, (w//self.down_factor, h//self.down_factor), interpolation=cv2.INTER_AREA)
        gt = torch.tensor(gt).float()
        gt = (gt - 127.5) / 127.5
        gt = gt.permute(2, 0, 1)    # (c, h, w)

        mask = self.masks[frame_idx]
        mask = cv2.resize(mask, (w//self.down_factor, h//self.down_factor), interpolation=cv2.INTER_AREA)
        mask = torch.from_numpy(mask).float()
        meas = torch.from_numpy(meas).permute(2, 0, 1)  # (c h w)
        cond = {'mask': mask, 'measure': meas_pixel_values}

        return gt, cond
    
def rgb2raw(block_size):
    r = np.array([[1, 0], [0, 0]])
    g1 = np.array([[0, 1], [0, 0]])
    g2 = np.array([[0, 0], [1, 0]])
    b = np.array([[0, 0], [0, 1]])
    rgb2raw = np.zeros([block_size, block_size, 3])
    rgb2raw[:, :, 0] = np.tile(r, (block_size // 2, block_size // 2))
    rgb2raw[:, :, 1] = np.tile(g1, (block_size // 2, block_size // 2)) + np.tile(g2, (
        block_size // 2, block_size // 2))
    rgb2raw[:, :, 2] = np.tile(b, (block_size // 2, block_size // 2))
    return rgb2raw
    
class RGBSingleFrameDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        down_factor: int = 4
    ):
        super().__init__()

        self.root = root
        self.split = split
        self.gt_paths = glob.glob(
            os.path.join(root, split, '*', '*', '*.png')
        )
        # self.masks = scio.loadmat('./mask.mat')
        self.masks = np.load('data/masks_reso64.npy')
        mask_s = self.masks.sum(axis=0)
        index = np.where(mask_s == 0)
        mask_s[index] = 1
        self.mask_s = mask_s.astype(np.float32)

        self.down_factor = down_factor
        self.rgb2raw = rgb2raw(64)

    def __len__(self):
        return len(self.gt_paths)
    
    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        data_dir, gt_name = os.path.split(gt_path)
        frame_idx = eval(gt_name.split('.')[0])
        

        # Load, tensorize and normalize
        gt = imageio.imread(gt_path)
        # Downsample
        h, w, c = gt.shape
        gt = cv2.resize(gt, (64, 64), interpolation=cv2.INTER_AREA)
        gt = torch.tensor(gt).float() / 255.
        # gt = (gt - 127.5) / 127.5
        gt = gt.permute(2, 0, 1)    # (c, h, w)

        # Mask
        mask = self.masks[frame_idx]
        mask = torch.from_numpy(mask).float()
        
        # Normalize measure
        meas_path = os.path.join(data_dir, 'measure_reso64.npy')
        if not os.path.exists(meas_path):
            frames = []
            for i in range(8):
                frame = imageio.imread(os.path.join(data_dir, f'{i}.png'))
                frame = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
                frames.append(frame)
            frames = np.stack(frames)
            meas = self.rgb_to_measurement(frames)
            np.save(meas_path, meas)
        else:
            meas = np.load(meas_path)
        meas = meas / self.mask_s[..., None]
        meas = meas / 255.0
        meas = torch.from_numpy(meas).float().permute(2, 0, 1)  # (c h w)
        coarse_est = meas * mask
        
        # cond = {'mask': mask, 'measure': meas_pixel_values}

        return gt, coarse_est
    

    def rgb_to_measurement(self, frames: np.ndarray):
        """
        frames, np.ndarray[dtype=np.uint8], (N, H, W, C)
        """
        meas = frames * self.rgb2raw
        meas = np.sum(meas * self.masks[..., None], axis=0)
        return meas
    



class RGBVideoDataset(Dataset):

    def __init__(
        self,
        root,
        split,
        video_length: int = 8,
    ):
        super().__init__()

        self.root = root
        self.split = split
        self.video_paths = glob.glob(
            os.path.join(root, split, '*', 'crop*')
        )
        # self.masks = scio.loadmat('./mask.mat')
        self.masks = np.load('data/masks_reso64.npy')
        mask_s = self.masks.sum(axis=0)
        index = np.where(mask_s == 0)
        mask_s[index] = 1
        self.mask_s = mask_s.astype(np.float32)

        self.rgb2raw = rgb2raw(64)
        self.video_length = video_length

        assert video_length == len(self.masks)

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        video_path = self.video_paths[index]

        # Load frames
        frames = []
        for i in range(self.video_length):
            gt_path = os.path.join(video_path, f'{i}.png')
            gt = imageio.imread(gt_path)
            # Downsample
            # h, w, c = gt.shape
            gt = cv2.resize(gt, (64, 64), interpolation=cv2.INTER_AREA)
            # gt = torch.tensor(gt).float() / 255.
            frames.append(gt)
        frames = np.stack(frames)
        
        # Normalize measure
        meas_path = os.path.join(video_path, 'measure_reso64.npy')
        if not os.path.exists(meas_path):
            meas = self.rgb_to_measurement(frames)
            np.save(meas_path, meas)
        else:
            meas = np.load(meas_path)
        meas = self.normalize_measurement(meas)
        meas = meas / 255.0
        meas = torch.from_numpy(meas).float().permute(2, 0, 1)  # (c h w)
        coarse_ests = meas * torch.from_numpy(self.masks).unsqueeze(1)  # (N, C, H, W)

        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2) # (N, C, H, W)
        
        # cond = {'mask': mask, 'measure': meas_pixel_values}

        return frames, coarse_ests

    def rgb_to_measurement(self, frames: np.ndarray):
        """
        frames, np.ndarray[dtype=np.uint8], (N, H, W, C)
        """
        meas = frames * self.rgb2raw
        meas = np.sum(meas * self.masks[..., None], axis=0).astype(np.float64)
        return meas
    
    def normalize_measurement(self, meas):
        return meas / self.mask_s[..., None]

        