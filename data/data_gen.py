"""
A python and RGB version of BIRNAT/train/data_generation.m
"""

import os
import numpy as np
import imageio.v2 as imageio
from tqdm.auto import tqdm, trange

root = './DAVIS'
save_path = './train'
reso = '480p'
block_size = 256
num_frame = 8
num_vid_per_obj = 5


def main():

    gt_dir = os.path.join(root, 'JPEGImages', reso)
    name_objs = os.listdir(gt_dir)

    os.makedirs(save_path, exist_ok=True)

    masks = np.load('data/masks.npy')

    # RGB2Raw
    r = np.array([[1, 0], [0, 0]])
    g1 = np.array([[0, 1], [0, 0]])
    g2 = np.array([[0, 0], [1, 0]])
    b = np.array([[0, 0], [0, 1]])
    rgb2raw = np.zeros([block_size, block_size, 3])
    rgb2raw[:, :, 0] = np.tile(r, (block_size // 2, block_size // 2))
    rgb2raw[:, :, 1] = np.tile(g1, (block_size // 2, block_size // 2)) + np.tile(g2, (
        block_size // 2, block_size // 2))
    rgb2raw[:, :, 2] = np.tile(b, (block_size // 2, block_size // 2))
    # rgb2raw = torch.from_numpy(rgb2raw).cuda().float()

    for obj in tqdm(name_objs, desc="Object"):
        save_path_ = os.path.join(save_path, obj)
        os.makedirs(save_path_, exist_ok=True)
        gt_dir_ = os.path.join(gt_dir, obj)
        gt_names = os.listdir(gt_dir_)
        img1 = imageio.imread(os.path.join(gt_dir_, gt_names[0]))
        h, w, c = img1.shape

        x = np.arange(0, w-block_size, block_size//2)
        if x[-1] < w - block_size:
            x[-1] = w - block_size
        y = np.arange(0, h-block_size, block_size//2)
        if y[-1] < h - block_size:
            y[-1] = h - block_size

        num_all_frame = len(gt_names)
        crop_counter = 0
        for f_start in trange(num_all_frame - num_frame + 1, desc="Frame start"):
            video_block_gt = []
            for f in range(num_frame):
                img = imageio.imread(os.path.join(gt_dir_, gt_names[f_start+f]))
                video_block_gt.append(img)
            video_block_gt = np.stack(video_block_gt)
            mean_vid_gt = np.mean(video_block_gt, axis=0)
            d_vid_gt = np.std(video_block_gt, axis=0)
            d_vid_gt = np.mean(d_vid_gt, axis=-1)
            m = np.zeros((len(y), len(x)))
            for i in range(len(y)):
                for j in range(len(x)):
                    x1 = d_vid_gt[y[i]:y[i]+block_size, x[j]:x[j]+block_size]
                    a1 = x1.max() - x1.min()
                    m[i, j] = a1
            m = m.flatten()
            indices = np.argsort(-m)[:num_vid_per_obj]
            for idx in indices:
                os.makedirs(os.path.join(save_path_, f'crop{crop_counter}'), exist_ok=True)
                i = idx % len(x); j = idx // len(x)
                x_start = x[i]; y_start = y[j]
                video_block = video_block_gt[:, y_start:y_start+block_size, x_start:x_start+block_size, :]
                for f in range(num_frame):
                    patch = video_block[f]
                    imageio.imwrite(os.path.join(save_path_, f'crop{crop_counter}', f'{f}.png'), patch)
                raw_block = video_block * rgb2raw
                meas = np.sum(raw_block, axis=0)
                np.save(os.path.join(save_path_, f'crop{crop_counter}', 'measure.npy'), meas)

                crop_counter += 1

if __name__ == '__main__':
    main()

        








