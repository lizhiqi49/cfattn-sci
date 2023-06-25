import os
import json
import math
import inspect
import logging
import imageio
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from PIL import Image
from tqdm.auto import tqdm
from typing import Optional, Literal, Union
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from einops import rearrange
from torchvision.utils import make_grid

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import transformers

import diffusers
# from diffusers import UNet2DConditionModel, DDIMScheduler
from diffusers.utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler

from data.dataset import RGBSingleFrameDataset
from models.unet import SciUNet2DModel


logger = get_logger(__name__, log_level="INFO")


# Copied from transformers.training_utils.SchedulerType
# For explicitely enum scheduler types
class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main(
    exp_name: str,
    unet_config_path: str,
    pretrained_unet_path: Optional[str] = None,
    seed: Optional[int] = 0,
    train_batch_size: int = 1,
    val_batch_size: int = 1,
    num_workers: int = 1,
    learning_rate: float = 1e-3,
    max_train_steps: int = 1000,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-3,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    scheduler_type: Union[str, SchedulerType] = 'constant',
    num_warmup_steps: Optional[int] = None,
    mixed_precision: Literal['no', 'fp16'] = 'fp16',
    gradient_accumulation_steps: int = 1,
    checkpointing_step_interv: int = 1000,
    validation_step_interv: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    output_dir: str = './output',

):  
    # ====== Experiment setup ====== #
    *_, config = inspect.getargvalues(inspect.currentframe())
    output_dir = os.path.join(output_dir, exp_name)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(output_dir, 'logs')
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/validation", exist_ok=True)
        os.makedirs(f"{output_dir}/pretrained", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))


    # ====== Model, dataset, optimizer, scheduler ====== #
    # Initialize model
    unet_config = load_config(unet_config_path)
    if pretrained_unet_path is not None:
        unet = SciUNet2DModel.from_pretrained(pretrained_unet_path)
        print(f"Start from {pretrained_unet_path}")
    else:
        unet = SciUNet2DModel.from_config(unet_config)

    # Grad and dtype setup
    weight_dtype = torch.float16 if accelerator.mixed_precision == 'fp16' else torch.float32
    unet.to(accelerator.device)

    # Dataset
    train_dset = RGBSingleFrameDataset(
        root="./dataset",
        split="train",
    )
    train_dloader = DataLoader(
        train_dset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
    )
    val_dset = RGBSingleFrameDataset(
        root='./dataset',
        split="train",
    )
    val_dloader = DataLoader(
        val_dset, batch_size=val_batch_size,
    ) 

    # Optimizer
    params = unet.parameters()
    optimizer = torch.optim.AdamW(
        params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon
    )

    # lr scheduler
    lr_scheduler = get_scheduler(
        scheduler_type, optimizer, 
        num_warmup_steps * accelerator.num_processes * accelerator.gradient_accumulation_steps, 
        max_train_steps * accelerator.num_processes * accelerator.gradient_accumulation_steps
    )

    # Wrap everything using accelerate
    unet, optimizer, train_dloader, val_dloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dloader, val_dloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(exp_name)

    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0
    
    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            if len(dirs) > 0:
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]
            else:
                path = None
        
        if path is None:
                accelerator.print(
                    f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                resume_from_checkpoint = False
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
            lr_scheduler.last_epoch = global_step
            
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                # if step % gradient_accumulation_steps == 0:
                #     progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Get input
                x, x_ce = batch
                x_pred = unet(x_ce).recons

                # Loss
                loss = F.mse_loss(x_pred.float(), x.float(), reduction='mean')

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step % checkpointing_step_interv == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % validation_step_interv == 0:
                    unet.eval()
                    if accelerator.is_main_process:

                        with torch.cuda.amp.autocast(dtype=weight_dtype):
                            for batch in val_dloader:    # only one batch
                                x, x_ce = batch
                                x_pred = unet(x_ce).recons
                                break

                            # post process
                            B = x.shape[0]
                            
                            grid = torch.cat([x_ce[:, None, ...], x[:, None, ...], x_pred[:, None, ...]], dim=1).cpu().float()
                            grid = rearrange(grid, 'b m c h w -> (b m) c h w')
                            grid = make_grid(grid, nrow=3,)
                            grid = rearrange(grid, 'c h w -> h w c')
                            grid = grid.numpy()
                            grid = (grid * 255).clip(0, 255).astype(np.uint8)
                            save_path = f"{output_dir}/validation/sample-{global_step}.png"
                            imageio.imwrite(save_path, grid)
                            logger.info(f"Saved samples to {save_path}")

                    unet.train()
                    

            if global_step >= max_train_steps:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(f"{output_dir}/pretrained")
        
    accelerator.end_training()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config)
    main(**config)




