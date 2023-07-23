import torch
import torchvision.transforms.functional as functional

from modules import devices, images, shared
from modules.processing import StableDiffusionProcessingTxt2Img

from ldm.modules.attention import BasicTransformerBlock

from scripts.marking import patch_process_sample, unmark_prompt_context


def encode_to_latent(p, image, w, h):
    image = images.resize_image(1, image, w, h)
    x = functional.pil_to_tensor(image)
    x = functional.center_crop(x, (w, h))  # just to be safe
    x = x.to(devices.device, dtype=devices.dtype_vae)
    x = ((x / 255.0) * 2.0 - 1.0).unsqueeze(0)

    # TODO: use caching to make this faster
    with devices.autocast():
        vae_output = p.sd_model.encode_first_stage(x)
        z = p.sd_model.get_first_stage_encoding(vae_output)
    return z.squeeze(0)


def get_latents_from_params(p, params, width, height):
    w, h = (width // 8) * 8, (height // 8) * 8
    w_latent, h_latent = width // 8, height // 8
    
    def get_latents(images, cached_latents=None):
        # check if latents need to be computed or recomputed (if image size changed e.g. due to high-res fix)
        if cached_latents is None:
            return [encode_to_latent(p, img, w, h) for img in images]
        else:
            ls = []
            for latent, img in zip(cached_latents, images):
                if latent.shape[-2:] != (w_latent, h_latent):
                    print(f"[FABRIC] Recomputing latent for image of size {img.size}")
                    latent = encode_to_latent(p, img, w, h)
                ls.append(latent)
            return ls
    
    params.pos_latents = get_latents(params.pos_images, params.pos_latents)
    params.neg_latents = get_latents(params.neg_images, params.neg_latents)
    return params.pos_latents, params.neg_latents


def patch_unet_forward_pass(p, unet, params):
    if not params.pos_images and not params.neg_images:
        print("[FABRIC] No images to found, aborting patching")
        return

    if not hasattr(unet, "_fabric_old_forward"):
        unet._fabric_old_forward = unet.forward

    null_ctx = p.sd_model.get_learned_conditioning([""])

    width = (p.width // 8) * 8
    height = (p.height // 8) * 8

    has_hires_fix = isinstance(p, StableDiffusionProcessingTxt2Img) and getattr(p, 'enable_hr', False)
    if has_hires_fix:
        if p.hr_resize_x == 0 and p.hr_resize_y == 0:
            hr_w = int(p.width * p.hr_scale)
            hr_h = int(p.height * p.hr_scale)
        else:
            hr_w, hr_h = p.hr_resize_x, p.hr_resize_y
        hr_w = (hr_w // 8) * 8
        hr_h = (hr_h // 8) * 8
    else:
        hr_h = width
        hr_w = height

    def new_forward(self, x, timesteps=None, context=None, **kwargs):
        _, uncond_ids, context = unmark_prompt_context(context)
        cond_ids = [i for i in range(context.size(0)) if i not in uncond_ids]
        has_cond = len(cond_ids) > 0
        has_uncond = len(uncond_ids) > 0

        w_latent, h_latent = x.shape[-2:]
        w, h = 8 * w_latent, 8 * h_latent
        if has_hires_fix and w == hr_w and h == hr_h:
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

        # fix for medvram option
        if shared.cmd_opts.medvram:
            try:
                # Trigger register_forward_pre_hook to move the model to correct device
                p.sd_model.model()
            except:
                pass

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
