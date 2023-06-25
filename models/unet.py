
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, List

import torch
import torch.nn as nn

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.attention_processor import AttentionProcessor, AttnProcessor

from .unet_blocks import (
    SciSelfAttnDownBlock2D,
    DownBlock2D,
    SciSelfAttnUpBlock2D,
    UpBlock2D,
    SciUNetMidBlock2DSelfAttn,
    get_down_block,
    get_up_block
)


@dataclass
class SciUNet2DOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states output. Output of last layer of model.
    """

    recons: torch.FloatTensor


class SciUNet2DModel(ModelMixin, ConfigMixin):
    r"""
    UNet2DModel is a 2D UNet model that takes in a noisy sample and a timestep and returns sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample. Dimensions must be a multiple of `2 ** (len(block_out_channels) -
            1)`.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input image.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D")`): Tuple of downsample block
            types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock2D"`):
            The mid block type. Choose from `UNetMidBlock2D` or `UnCLIPUNetMidBlock2D`.
        up_block_types (`Tuple[str]`, *optional*, defaults to :
            obj:`("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D")`): Tuple of upsample block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to :
            obj:`(224, 448, 672, 896)`): Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): The number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): The scale factor for the mid block.
        downsample_padding (`int`, *optional*, defaults to `1`): The padding for the downsample convolution.
        act_fn (`str`, *optional*, defaults to `"silu"`): The activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): The attention head dimension.
        norm_num_groups (`int`, *optional*, defaults to `32`): The number of groups for the normalization.
        norm_eps (`float`, *optional*, defaults to `1e-5`): The epsilon for the normalization.
    """

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        block_out_channels: Tuple[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        # add_attention: bool = True,
    ):
        super().__init__()

        self.sample_size = sample_size
        # time_embed_dim = block_out_channels[0] * 4

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = SciUNetMidBlock2DSelfAttn(
            in_channels=block_out_channels[-1],
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
            # add_attention=add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=attention_head_dim,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[SciUNet2DOutput, Tuple]:
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`: [`~models.unet_2d.UNet2DOutput`] if `return_dict` is True,
            otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.
        """

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, )

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, skip_sample)
            else:
                sample = upsample_block(sample, res_samples)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample


        if not return_dict:
            return (sample,)

        return SciUNet2DOutput(recons=sample)
    
    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Parameters:
            `processor (`dict` of `AttentionProcessor` or `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                of **all** `Attention` layers.
            In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainable attention processors.:

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(AttnProcessor())

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maximum amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_sliceable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_sliceable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_sliceable_dims(module)

        num_sliceable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_sliceable_layers * [1]

        slice_size = num_sliceable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (SciSelfAttnDownBlock2D, DownBlock2D, SciSelfAttnUpBlock2D, UpBlock2D)):
            module.gradient_checkpointing = value




