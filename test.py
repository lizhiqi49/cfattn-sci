import os
import argparse
import numpy as np
import cv2
import torch
import imageio.v3 as imageio
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

from models.unet import SciUNet2DModel
from data.dataset import RGBVideoDataset
from models.attention_processor import CrossFrameAttnProcessor, set_cross_frame_attn_processor


def unwrap_frames(frames: np.ndarray):
    frame_list = []
    for f in frames:
        frame_list.append(f)
    return frame_list

def save_to_video(save_path, frames):
    imageio.mimsave(save_path, frames, fps=4, quality=10)


def main(
    pretrained_unet_path: str,
    test_video_dir: str,
    use_cross_frame_attn: bool = True,
    video_length: int = 8,
    start_idx: int = 0,
    end_idx: int = -1,
    reso: int = 64,
    save_video: bool = False,
):
    device = torch.device('cuda')
    torch_dtype = torch.float16

    unet = SciUNet2DModel.from_pretrained(pretrained_unet_path)
    print(f"Load pretrained model from {pretrained_unet_path}.")
    unet.eval()
    unet.requires_grad_(False)
    unet.to(device, dtype=torch_dtype)

    if use_cross_frame_attn:
        cross_frame_attn_procs = CrossFrameAttnProcessor(video_length=video_length)
        set_cross_frame_attn_processor(
            unet, cross_frame_attn_procs, 
            torch_dtype=torch_dtype,
        )
        print("Using cross-frame attention.")

    dset = RGBVideoDataset(root='train', split='', video_length=video_length)

    frames = []
    for i in range(start_idx, start_idx+video_length):
        frame_path = os.path.join(test_video_dir, f'{i}.png')
        # frame = imageio.imread(frame_path)
        frame = Image.open(frame_path).convert('RGB')
        frame = np.array(frame)
        h, w, c = frame.shape
        if h != reso or w != reso:
            frame = cv2.resize(frame, (reso, reso))
        frames.append(frame)
    frames = np.stack(frames)
    print("All frames loaded.")

    meas = dset.rgb_to_measurement(frames)
    meas = dset.normalize_measurement(meas)
    meas = torch.from_numpy(meas).float().permute(2, 0, 1) / 255.0
    x_ce = meas * torch.from_numpy(dset.masks).unsqueeze(1)

    x_ce = x_ce.to(device, dtype=torch_dtype)
    with torch.no_grad():
        x_pred = unet(x_ce).recons
    print("Prediction finished.")

    if save_video:
        frames = x_pred.cpu().permute(0, 2, 3, 1).numpy().clip(0, 1) * 255.0
        frames = frames.astype(np.uint8)
        frames = unwrap_frames(frames)
        save_path = './test_result.mp4'
        save_to_video(save_path, frames)
    else:
        x = torch.from_numpy(frames).float().permute(0, 3, 1, 2).cuda() / 255.
        grid = torch.cat([x_ce, x_pred, x], dim=0).cpu().float()
        grid = make_grid(grid, nrow=video_length,)
        grid = rearrange(grid, 'c h w -> h w c')
        grid = grid.numpy()
        grid = (grid * 255).clip(0, 255).astype(np.uint8)
        save_path = './test_result.png'
        imageio.imwrite(save_path, grid)
    print(f"Test result saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_unet_path', type=str)
    parser.add_argument('--test_video_dir', type=str)
    parser.add_argument('--use_cross_frame_attn', action='store_true', default=False)
    parser.add_argument('--video_length', type=int, default=8)
    parser.add_argument('--save_video', action='store_true', default=False)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    args = parser.parse_args()

    main(**vars(args))
    
    


