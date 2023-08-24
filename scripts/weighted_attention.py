import math
import functools

import torch
import torch.nn.functional
from torch import einsum
from einops import rearrange

from ldm.util import default

import modules.sd_hijack_optimizations
from modules import shared, devices
from modules.hypernetworks import hypernetwork
from modules.sd_hijack_optimizations import (
    split_cross_attention_forward_invokeAI,
    xformers_attention_forward,
    scaled_dot_product_no_mem_attention_forward,
    scaled_dot_product_attention_forward,
    split_cross_attention_forward,
    get_available_vram,
)


try:
    import xformers.ops
    _xformers_attn = xformers.ops.memory_efficient_attention
except ImportError:
    pass

_einsum_op_compvis = modules.sd_hijack_optimizations.einsum_op_compvis
_sdp_attention = torch.nn.functional.scaled_dot_product_attention


def patched_einsum_op_compvis(q, k, v, weights=None):
    s = einsum('b i d, b j d -> b i j', q, k)
    s = s.softmax(dim=-1, dtype=s.dtype)
    if weights is not None:
        s = s * weights[None, None, :]
    return einsum('b i j, b j d -> b i d', s, v)


def patched_xformers_attn(q, k, v, attn_bias=None, op=None, weights=None, orig_attn=None):
    bs, nq, nh, dh = q.shape  # batch_size, num_queries, num_heads, dim_per_head
    if weights is not None:
        min_val = torch.finfo(q.dtype).min
        w_bias = weights.log().clamp(min=min_val)[None, None, None, :].expand(bs, nh, nq, -1).contiguous()
        w_bias = w_bias.to(dtype=q.dtype)
        if attn_bias is None:
            attn_bias = w_bias
        else:
            attn_bias += w_bias
    return orig_attn(q, k, v, attn_bias=attn_bias, op=op)


def patched_sdp_attn(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, weights=None, orig_attn=None):
    if attn_mask is not None:
        attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
        attn_mask = attn_mask.to(dtype=q.dtype)
    if weights is not None:
        min_val = torch.finfo(q.dtype).min
        w_bias = weights.log().clamp(min=min_val)[None, None, None, :].expand(*q.shape[:3], -1)
        if attn_mask is None:
            attn_mask = w_bias
        else:
            attn_mask += w_bias
    return orig_attn(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)


# copied and adapted from modules.sd_hijack_optimizations.split_cross_attention_forward
def weighted_split_cross_attention_forward(self, x, context=None, mask=None, weights=None):
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    dtype = q_in.dtype
    if shared.opts.upcast_attn:
        q_in, k_in, v_in = q_in.float(), k_in.float(), v_in if v_in.device.type == 'mps' else v_in.float()

    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        k_in = k_in * self.scale

        del context, x

        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q_in, k_in, v_in))
        del q_in, k_in, v_in

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

        mem_free_total = get_available_vram()

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier

        # FABRIC some batch-size dependend overhead. Found empirically on RTX 3090.
        bs = q.shape[0] / 8  # batch size
        mem_required *= 1/(bs + 1) + 1.25
        mem_required *= 1.05  # safety margin
        steps = 1

        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
            # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
            #       f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)

            # OURS: apply weights to attention
            if weights is not None:
                bias = weights.to(s1.dtype).log().clamp(min=torch.finfo(s1.dtype).min)
                s1 = s1 + bias
                del bias

            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1

            r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            del s2

        del q, k, v

    r1 = r1.to(dtype)

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)


def is_the_same(fn1, fn2):
    if isinstance(fn2, (list, tuple)):
        return any(is_the_same(fn1, f) for f in fn2)
    return fn1.__name__ == fn2.__name__ and fn1.__module__ == fn2.__module__


def weighted_attention(self, attn_fn, x, context=None, weights=None, **kwargs):
    if weights is None:
        return attn_fn(x, context=context, **kwargs)

    if is_the_same(attn_fn, split_cross_attention_forward_invokeAI):
        modules.sd_hijack_optimizations.einsum_op_compvis = functools.partial(patched_einsum_op_compvis, weights=weights)
        out = attn_fn(x, context=context, **kwargs)
        modules.sd_hijack_optimizations.einsum_op_compvis = _einsum_op_compvis
        return out
    
    elif is_the_same(attn_fn, xformers_attention_forward):
        assert _xformers_attn in locals() or _xformers_attn in globals(), "xformers attention function not found"
        xformers.ops.memory_efficient_attention = functools.partial(patched_xformers_attn, weights=weights, orig_attn=_xformers_attn)
        out = attn_fn(x, context=context, **kwargs)
        xformers.ops.memory_efficient_attention = _xformers_attn
        return out
    
    elif is_the_same(attn_fn, [scaled_dot_product_no_mem_attention_forward, scaled_dot_product_attention_forward]):
        torch.nn.functional.scaled_dot_product_attention = functools.partial(patched_sdp_attn, weights=weights, orig_attn=_sdp_attention)
        out = attn_fn(x, context=context, **kwargs)
        torch.nn.functional.scaled_dot_product_attention = _sdp_attention
        return out
    
    elif is_the_same(attn_fn, split_cross_attention_forward):
        return weighted_split_cross_attention_forward(self, x, context=context, weights=weights, **kwargs)
    
    else:
        raise NotImplementedError(f"FABRIC does not support `{attn_fn.__module__}.{attn_fn.__name__}` yet. Please choose a supported attention function.")
