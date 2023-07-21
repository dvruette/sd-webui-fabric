import torch

from ldm.modules.attention import BasicTransformerBlock


def use_feedback(params):
    if not params.enabled:
        return False
    if params.start >= params.end and params.min_weight <= 0:
        return False
    if params.max_weight <= 0:
        return False
    if params.neg_scale <= 0 and len(params.pos_latents) == 0:
        return False
    if len(params.pos_latents) == 0 and len(params.neg_latents) == 0:
        return False
    return True

def patch_unet_forward_pass(p, unet, params):
    if not use_feedback(params):
        return
    
    if not hasattr(unet, "_fabric_old_forward"):
        unet._fabric_old_forward = unet.forward

    batch_size = p.batch_size

    def new_forward(self, x, timesteps=None, context=None, **kwargs):
        # save original forward pass
        for module in self.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1._fabric_old_forward = module.attn1.forward

        # add noise to reference latents
        all_latents = torch.stack(params.pos_latents + params.neg_latents, dim=0)
        all_zs = p.sd_model.q_sample(all_latents, torch.round(timesteps.float()).long())
        # TODO: confirm that slicing like this is correct for separating cond/uncond batch
        all_zs = all_zs[:all_zs.size(0) // 2]

        ## cache hidden states
        cached_hiddens = []
        def patched_attn1_forward(attn1, x, **kwargs):
            cached_hiddens.append(x.detach().clone())
            out = attn1._fabric_old_forward(x, **kwargs)
            return out

        # patch forward pass to cache hidden states
        for module in self.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = patched_attn1_forward.__get__(module.attn1)

        # run forward pass just to cache hidden states, output is discarded
        all_zs = all_zs.to(x.device, dtype=self.dtype)
        ts = timesteps[:1].expand(all_zs.size(0))  # (n_pos + n_neg,)
        # TODO: instead of using the negative prompt, use the null prompt
        ctx = context[batch_size:][:1].expand(all_zs.size(0), -1, -1)  # (n_pos + n_neg, p_seq, p_dim)
        _ = self._fabric_old_forward(all_zs, ts, ctx)

        def patched_attn1_forward(attn1, x, context=None, **kwargs):
            if context is None:
                context = x

            cached_hs = cached_hiddens.pop(0)

            seq_len, d_model = x.shape[1:]
            num_pos = len(params.pos_latents)
            num_neg = len(params.neg_latents)

            pos_hs = cached_hs[:num_pos].view(1, num_pos * seq_len, d_model).expand(batch_size, -1, -1)  # (bs, seq * n_pos, dim)
            neg_hs = cached_hs[num_pos:].view(1, num_neg * seq_len, d_model).expand(batch_size, -1, -1)  # (bs, seq * n_neg, dim)

            x_cond = x[:batch_size]  # (bs, seq, dim)
            x_uncond = x[batch_size:]  # (bs, seq, dim)
            ctx_cond = torch.cat([context[:batch_size], pos_hs], dim=1)  # (bs, seq * (1 + n_pos), dim)
            ctx_uncond = torch.cat([context[batch_size:], neg_hs], dim=1)  # (bs, seq * (1 + n_neg), dim)

            out_cond = attn1._fabric_old_forward(x_cond, ctx_cond, **kwargs)  # (bs, seq, dim)
            out_uncond = attn1._fabric_old_forward(x_uncond, ctx_uncond, **kwargs)  # (bs, seq, dim)
            out = torch.cat([out_cond, out_uncond], dim=0)
            return out

        # patch forward pass to inject cached hidden states
        for module in self.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = patched_attn1_forward.__get__(module.attn1)

        # run forward pass with cached hidden states
        out = self._fabric_old_forward(x, timesteps, context, **kwargs)

        # restore original pass
        for module in self.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1.forward = module.attn1._fabric_old_forward
                del module.attn1._fabric_old_forward

        return out
    
    unet.forward = new_forward.__get__(unet)

def unpatch_unet_forward_pass(unet):
    if hasattr(unet, "_fabric_old_forward"):
        unet.forward = unet._fabric_old_forward
        del unet._fabric_old_forward
