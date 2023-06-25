import torch
import xformers
from typing import Optional, Callable, Union
from einops import rearrange, repeat
from diffusers.models.attention_processor import Attention

class CrossFrameAttnProcessor:
    
    def __init__(self, video_length):
        self.video_length = video_length
    
    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        video_length = self.video_length
        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0; former_frame_index[-1] = video_length - 1
        later_frame_index = torch.arange(video_length) + 1
        later_frame_index[0] = 0; later_frame_index[-1] = video_length - 1

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        # key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
        key = torch.cat([key[:, former_frame_index], key[:, later_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        # value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
        value = torch.cat([value[:, former_frame_index], value[:, later_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    
    
class XFormersCrossFrameAttnProcessor:
    def __init__(self, video_length, attention_op: Optional[Callable] = None):
        self.video_length = video_length
        self.attention_op = attention_op

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        
        video_length = self.video_length
        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def set_cross_frame_attn_processor(
    model, 
    processor: Union[CrossFrameAttnProcessor, XFormersCrossFrameAttnProcessor],
    requires_grad: bool = False,
    torch_dtype: torch.dtype = torch.float32,
    return_params: bool = False
):
    r"""
    Parameters:
        `processor (`dict` of `AttentionProcessor` or `AttentionProcessor`):
            The instantiated processor class or a dictionary of processor classes that will be set as the processor
            of **all** `Attention` layers.
        In case `processor` is a dict, the key needs to define the path to the corresponding cross attention processor. This is strongly recommended when setting trainable attention processors.:

    """
    params = [] if return_params else None

    def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor, params=None):
        if hasattr(module, "set_processor") and name.endswith('attn2'):
            module.set_processor(processor)
            module.requires_grad_(requires_grad)
            module.to(dtype=torch_dtype)
            if params is not None:
                params += list(module.parameters())

        for sub_name, child in module.named_children():
            fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor, params)

    for name, module in model.named_children():
        fn_recursive_attn_processor(name, module, processor, params)

    if return_params:
        return params