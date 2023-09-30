import math
import psutil

import torch
import torch.nn.functional
from torch import einsum
from einops import rearrange

from ldm.util import default

from modules import shared, devices, sd_hijack
from modules.hypernetworks import hypernetwork
from modules.sd_hijack_optimizations import (
    get_xformers_flash_attention_op,
    get_available_vram,
)

try:
    import xformers
    import xformers.ops
except ImportError:
    pass


def get_weighted_attn_fn():
    method = sd_hijack.model_hijack.optimization_method
    if method is None:
        return weighted_split_cross_attention_forward
    method = method.lower()

    if method not in ['none', 'sdp-no-mem', 'sdp', 'xformers', 'sub-quadratic', 'v1', 'invokeai', 'doggettx']:
        print(f"[FABRIC] Warning: Unknown attention optimization method {method}.")
        return weighted_split_cross_attention_forward
    
    if method == 'none':
        return weighted_split_cross_attention_forward
    elif method == 'xformers':
        return weighted_xformers_attention_forward
    elif method == 'sdp-no-mem':
        return weighted_scaled_dot_product_no_mem_attention_forward
    elif method == 'sdp':
        return weighted_scaled_dot_product_attention_forward
    elif method == 'doggettx':
        return weighted_split_cross_attention_forward
    elif method == 'invokeai':
        return weighted_split_cross_attention_forward_invokeAI
    elif method == 'sub-quadratic':
        print(f"[FABRIC] Warning: Sub-quadratic attention is not supported yet. Please open an issue if you need this for your workflow. Falling back to split attention.")
        return weighted_split_cross_attention_forward
    elif method == 'v1':
        print(f"[FABRIC] Warning: V1 attention is not supported yet. Please open an issue if you need this for your workflow. Falling back to split attention.")
        return weighted_split_cross_attention_forward
    else:
        return weighted_split_cross_attention_forward


def weighted_attention(self, attn_fn, x, context=None, weights=None, **kwargs):
    if weights is None:
        return attn_fn(x, context=context, **kwargs)
    
    weighted_attn_fn = get_weighted_attn_fn()
    return weighted_attn_fn(self, x, context=context, weights=weights, **kwargs)


def _get_attn_bias(weights, shape=None, dtype=torch.float32):
    # shape of weights needs to be divisible by 8 in order for xformers attn bias to work
    last_dim = ((weights.shape[-1] - 1) // 8 + 1) * 8
    w_bias = torch.zeros(weights.shape[:-1] + (last_dim,), device=weights.device, dtype=weights.dtype)
    
    min_val = torch.finfo(dtype).min
    w_bias[..., :weights.shape[-1]] = weights.log().clamp(min=min_val)
    
    if shape is not None:
        assert shape[-1] == weights.shape[-1], "Last dimension of shape must match last dimension of weights (number of keys)"
        w_bias = w_bias.view([1] * (len(shape) - 1) + [-1]).expand(shape[:-1] + (last_dim,))

    # cast first in order to preserve multiple-of-8 stride
    w_bias = w_bias.to(dtype=dtype)
    w_bias = w_bias[..., :weights.shape[-1]]
    return w_bias

### The following attn functions are copied and adapted from modules.sd_hijack_optimizations

# --- InvokeAI ---
mem_total_gb = psutil.virtual_memory().total // (1 << 30)

def einsum_op_compvis(q, k, v, weights=None):
    s = einsum('b i d, b j d -> b i j', q, k)
    if weights is not None:
        s += _get_attn_bias(weights, s.shape, s.dtype)
    s = s.softmax(dim=-1, dtype=s.dtype)
    return einsum('b i j, b j d -> b i d', s, v)

def einsum_op_slice_0(q, k, v, slice_size, weights=None):
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[0], slice_size):
        end = i + slice_size
        r[i:end] = einsum_op_compvis(q[i:end], k[i:end], v[i:end], weights)
    return r

def einsum_op_slice_1(q, k, v, slice_size, weights=None):
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        r[:, i:end] = einsum_op_compvis(q[:, i:end], k, v, weights)
    return r

def einsum_op_mps_v1(q, k, v, weights=None):
    if q.shape[0] * q.shape[1] <= 2**16: # (512x512) max q.shape[1]: 4096
        return einsum_op_compvis(q, k, v, weights)
    else:
        slice_size = math.floor(2**30 / (q.shape[0] * q.shape[1]))
        if slice_size % 4096 == 0:
            slice_size -= 1
        return einsum_op_slice_1(q, k, v, slice_size, weights)

def einsum_op_mps_v2(q, k, v, weights=None):
    if mem_total_gb > 8 and q.shape[0] * q.shape[1] <= 2**16:
        return einsum_op_compvis(q, k, v, weights)
    else:
        return einsum_op_slice_0(q, k, v, 1, weights)

def einsum_op_tensor_mem(q, k, v, max_tensor_mb, weights=None):
    size_mb = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size() // (1 << 20)
    if size_mb <= max_tensor_mb:
        return einsum_op_compvis(q, k, v, weights)
    div = 1 << int((size_mb - 1) / max_tensor_mb).bit_length()
    if div <= q.shape[0]:
        return einsum_op_slice_0(q, k, v, q.shape[0] // div, weights)
    return einsum_op_slice_1(q, k, v, max(q.shape[1] // div, 1), weights)

def einsum_op_cuda(q, k, v, weights=None):
    stats = torch.cuda.memory_stats(q.device)
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(q.device)
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch
    # Divide factor of safety as there's copying and fragmentation
    return einsum_op_tensor_mem(q, k, v, mem_free_total / 3.3 / (1 << 20), weights)

def einsum_op(q, k, v, weights=None):
    if q.device.type == 'cuda':
        return einsum_op_cuda(q, k, v, weights)

    if q.device.type == 'mps':
        if mem_total_gb >= 32 and q.shape[0] % 32 != 0 and q.shape[0] * q.shape[1] < 2**18:
            return einsum_op_mps_v1(q, k, v, weights)
        return einsum_op_mps_v2(q, k, v, weights)

    # Smaller slices are faster due to L2/L3/SLC caches.
    # Tested on i7 with 8MB L3 cache.
    return einsum_op_tensor_mem(q, k, v, 32, weights)

def weighted_split_cross_attention_forward_invokeAI(self, x, context=None, mask=None, weights=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k = self.to_k(context_k)
    v = self.to_v(context_v)
    del context, context_k, context_v, x

    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v if v.device.type == 'mps' else v.float()

    with devices.without_autocast(disable=not shared.opts.upcast_attn):
        k = k * self.scale

        q, k, v = (rearrange(t, 'b n (h d) -> (b h) n d', h=h) for t in (q, k, v))
        r = einsum_op(q, k, v, weights)
    r = r.to(dtype)
    return self.to_out(rearrange(r, '(b h) n d -> b n (h d)', h=h))
# --- end InvokeAI ---


def weighted_xformers_attention_forward(self, x, context=None, mask=None, weights=None):

    h = self.heads
    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    q, k, v = (rearrange(t, 'b n (h d) -> b n h d', h=h) for t in (q_in, k_in, v_in))
    del q_in, k_in, v_in

    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()

    ### FABRIC ###
    bias_shape = (q.size(0), q.size(2), q.size(1), k.size(1))  # (bs, h, nq, nk)
    if weights is not None:
        attn_bias = _get_attn_bias(weights, bias_shape, dtype=q.dtype)
    else:
        attn_bias = None
    ### END FABRIC ###

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias, op=get_xformers_flash_attention_op(q, k, v))

    out = out.to(dtype)

    out = rearrange(out, 'b n h d -> b n (h d)', h=h)
    return self.to_out(out)


def weighted_scaled_dot_product_attention_forward(self, x, context=None, mask=None, weights=None):
    batch_size, sequence_length, inner_dim = x.shape

    if mask is not None:
        mask = self.prepare_attention_mask(mask, sequence_length, batch_size)
        mask = mask.view(batch_size, self.heads, -1, mask.shape[-1])

    h = self.heads
    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(shared.loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    head_dim = inner_dim // h
    q = q_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    k = k_in.view(batch_size, -1, h, head_dim).transpose(1, 2)
    v = v_in.view(batch_size, -1, h, head_dim).transpose(1, 2)

    del q_in, k_in, v_in

    dtype = q.dtype
    if shared.opts.upcast_attn:
        q, k, v = q.float(), k.float(), v.float()

    ### FABRIC ###
    mask_shape = q.shape[:3] + (k.shape[2],)  # (bs, h, nq, nk)
    if mask is None:
        mask = 0
    else:
        mask.masked_fill(not mask, -float('inf')) if mask.dtype==torch.bool else mask
        mask = mask.to(dtype=q.dtype)
    if weights is not None:
        w_bias = _get_attn_bias(weights, mask_shape, dtype=q.dtype)
        mask += w_bias
    ### END FABRIC ###

    # the output of sdp = (batch, num_heads, seq_len, head_dim)
    hidden_states = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, h * head_dim)
    hidden_states = hidden_states.to(dtype)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)
    return hidden_states

def weighted_scaled_dot_product_no_mem_attention_forward(self, x, context=None, mask=None, weights=None):
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=False):
        return weighted_scaled_dot_product_attention_forward(self, x, context, mask, weights)


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

        # FABRIC incurs some batch-size-dependend overhead. Found empirically on RTX 3090.
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
