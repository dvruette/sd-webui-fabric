import torch
import torchvision.transforms.functional as functional

from modules import devices, images

from ldm.modules.attention import BasicTransformerBlock

from scripts.marking import patch_process_sample, unmark_prompt_context


def encode_to_latent(p, image, size):
    w, h = size
    image = images.resize_image(1, image, w, h)
    x = functional.pil_to_tensor(image)
    x = functional.center_crop(x, (h, w))  # just to be safe
    x = x.to(devices.device, dtype=devices.dtype_vae)
    x = ((x / 255.0) * 2.0 - 1.0).unsqueeze(0)

    # TODO: use caching to make this faster
    with devices.autocast():
        vae_output = p.sd_model.encode_first_stage(x)
        z = p.sd_model.get_first_stage_encoding(vae_output)
    return z.squeeze(0)


def get_latents_from_params(p, params, width, height):
    w_latent, h_latent = width // 8, height // 8
    # check if latents need to be computed or recomputed (if image size changed e.g. due to high-res fix)
    if params.pos_latents is None:
        pos_latents = [encode_to_latent(p, img, (width, height)) for img in params.pos_images]
    else:
        pos_latents = []
        for latent in params.pos_latents:
            if latent.shape[-2:] != (w_latent, h_latent):
                latent = images.resize_image(1, latent, width, height)
            pos_latents.append(latent)
        params.pos_latents = pos_latents
    # do the same for negative latents
    if params.neg_latents is None:
        neg_latents = [encode_to_latent(p, img, (width, height)) for img in params.neg_images]
    else:
        neg_latents = []
        for latent in params.neg_latents:
            if latent.shape[-2:] != (w_latent, h_latent):
                latent = images.resize_image(1, latent, width, height)
            neg_latents.append(latent)
        params.neg_latents = neg_latents
    return pos_latents, neg_latents


def patch_unet_forward_pass(p, unet, params):
    if not params.pos_images and not params.neg_images:
        return

    if not hasattr(unet, "_fabric_old_forward"):
        unet._fabric_old_forward = unet.forward

    null_ctx = p.sd_model.get_learned_conditioning([""])

    def new_forward(self, x, timesteps=None, context=None, **kwargs):
        _, uncond_ids, context = unmark_prompt_context(context)
        cond_ids = [i for i in range(context.size(0)) if i not in uncond_ids]
        has_cond = len(cond_ids) > 0
        has_uncond = len(uncond_ids) > 0

        w_latent, h_latent = x.shape[-2:]
        w, h = 8 * w_latent, 8 * h_latent
        if w != p.width or h != p.height:
            if not params.feedback_during_high_res_fix:
                return self._fabric_old_forward(x, timesteps, context, **kwargs)

        pos_latents, neg_latents = get_latents_from_params(p, params, w, h)
        pos_latents = pos_latents if has_cond else []
        neg_latents = neg_latents if has_uncond else []
        all_latents = pos_latents + neg_latents
        if len(all_latents) == 0:
            return self._fabric_old_forward(x, timesteps, context, **kwargs)

        # add noise to reference latents
        all_zs = []
        for latent in all_latents:
            z = p.sd_model.q_sample(latent.unsqueeze(0), torch.round(timesteps.float()).long())[0]
            all_zs.append(z)
        all_zs = torch.stack(all_zs, dim=0)

        # save original forward pass
        for module in self.modules():
            if isinstance(module, BasicTransformerBlock):
                module.attn1._fabric_old_forward = module.attn1.forward

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
        # use the null prompt for pre-computing hidden states on feedback images
        ctx = null_ctx.expand(all_zs.size(0), -1, -1)  # (n_pos + n_neg, p_seq, p_dim)
        _ = self._fabric_old_forward(all_zs, ts, ctx)

        def patched_attn1_forward(attn1, x, context=None, **kwargs):
            if context is None:
                context = x

            cached_hs = cached_hiddens.pop(0)

            seq_len, d_model = x.shape[1:]
            num_pos = len(pos_latents)
            num_neg = len(neg_latents)
            num_cond = len(cond_ids)
            num_uncond = len(uncond_ids)

            outs = []
            if num_cond > 0:
                pos_hs = cached_hs[:num_pos].view(1, num_pos * seq_len, d_model).expand(num_cond, -1, -1)  # (n_cond, seq * n_pos, dim)
                x_cond = x[cond_ids]  # (n_cond, seq, dim)
                ctx_cond = torch.cat([context[cond_ids], pos_hs], dim=1)  # (n_cond, seq * (1 + n_pos), dim)
                out_cond = attn1._fabric_old_forward(x_cond, ctx_cond, **kwargs)  # (n_cond, seq, dim)
                outs.append(out_cond)
            if num_uncond > 0:
                neg_hs = cached_hs[num_pos:].view(1, num_neg * seq_len, d_model).expand(num_uncond, -1, -1)  # (n_uncond, seq * n_neg, dim)
                x_uncond = x[uncond_ids]  # (n_uncond, seq, dim)
                ctx_uncond = torch.cat([context[uncond_ids], neg_hs], dim=1)  # (n_uncond, seq * (1 + n_neg), dim)
                out_uncond = attn1._fabric_old_forward(x_uncond, ctx_uncond, **kwargs)  # (n_uncond, seq, dim)
                outs.append(out_uncond)
            out = torch.cat(outs, dim=0)
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

    patch_process_sample(p)

def unpatch_unet_forward_pass(unet):
    if hasattr(unet, "_fabric_old_forward"):
        unet.forward = unet._fabric_old_forward
        del unet._fabric_old_forward
