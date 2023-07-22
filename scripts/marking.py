import torch

from modules.prompt_parser import MulticondLearnedConditioning, ComposableScheduledPromptConditioning, ScheduledPromptConditioning


"""
We adopt the same marking strategy as ControlNet for determining whether a prompt is conditional or unconditional.
For the original implementation see: https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/hook.py
"""

POSITIVE_MARK_TOKEN = 1024
NEGATIVE_MARK_TOKEN = - POSITIVE_MARK_TOKEN
MARK_EPS = 1e-3


def process_sample(process, *args, **kwargs):
    # ControlNet must know whether a prompt is conditional prompt (positive prompt) or unconditional conditioning prompt (negative prompt).
    # You can use the hook.py's `mark_prompt_context` to mark the prompts that will be seen by ControlNet.
    # Let us say XXX is a MulticondLearnedConditioning or a ComposableScheduledPromptConditioning or a ScheduledPromptConditioning or a list of these components,
    # if XXX is a positive prompt, you should call mark_prompt_context(XXX, positive=True)
    # if XXX is a negative prompt, you should call mark_prompt_context(XXX, positive=False)
    # After you mark the prompts, the ControlNet will know which prompt is cond/uncond and works as expected.
    # After you mark the prompts, the mismatch errors will disappear.
    mark_prompt_context(kwargs.get('conditioning', []), positive=True)
    mark_prompt_context(kwargs.get('unconditional_conditioning', []), positive=False)
    mark_prompt_context(getattr(process, 'hr_c', []), positive=True)
    mark_prompt_context(getattr(process, 'hr_uc', []), positive=False)
    return process.sample_before_CN_hack(*args, **kwargs)


def prompt_context_is_marked(x):
    t = x[..., 0, :]
    m = torch.abs(t) - POSITIVE_MARK_TOKEN
    m = torch.mean(torch.abs(m)).detach().cpu().float().numpy()
    return float(m) < MARK_EPS


def mark_prompt_context(x, positive):
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = mark_prompt_context(x[i], positive)
        return x
    if isinstance(x, MulticondLearnedConditioning):
        x.batch = mark_prompt_context(x.batch, positive)
        return x
    if isinstance(x, ComposableScheduledPromptConditioning):
        x.schedules = mark_prompt_context(x.schedules, positive)
        return x
    if isinstance(x, ScheduledPromptConditioning):
        cond = x.cond
        if prompt_context_is_marked(cond):
            return x
        mark = POSITIVE_MARK_TOKEN if positive else NEGATIVE_MARK_TOKEN
        cond = torch.cat([torch.zeros_like(cond)[:1] + mark, cond], dim=0)
        return ScheduledPromptConditioning(end_at_step=x.end_at_step, cond=cond)
    return x


def unmark_prompt_context(x):
    if not prompt_context_is_marked(x):
        # ControlNet must know whether a prompt is conditional prompt (positive prompt) or unconditional conditioning prompt (negative prompt).
        # You can use the hook.py's `mark_prompt_context` to mark the prompts that will be seen by ControlNet.
        # Let us say XXX is a MulticondLearnedConditioning or a ComposableScheduledPromptConditioning or a ScheduledPromptConditioning or a list of these components,
        # if XXX is a positive prompt, you should call mark_prompt_context(XXX, positive=True)
        # if XXX is a negative prompt, you should call mark_prompt_context(XXX, positive=False)
        # After you mark the prompts, the ControlNet will know which prompt is cond/uncond and works as expected.
        # After you mark the prompts, the mismatch errors will disappear.
        mark_batch = torch.ones(size=(x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device)
        uc_indices = []
        context = x
        return mark_batch, uc_indices, context
    mark = x[:, 0, :]
    context = x[:, 1:, :]
    mark = torch.mean(torch.abs(mark - NEGATIVE_MARK_TOKEN), dim=1)
    mark = (mark > MARK_EPS).float()
    mark_batch = mark[:, None, None, None].to(x.dtype).to(x.device)
    uc_indices = mark.detach().cpu().numpy().tolist()
    uc_indices = [i for i, item in enumerate(uc_indices) if item < 0.5]
    return mark_batch, uc_indices, context


def patch_process_sample(process):
    if getattr(process, 'sample_before_CN_hack', None) is None:
        process.sample_before_CN_hack = process.sample
    process.sample = process_sample.__get__(process)
