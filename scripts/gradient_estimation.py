## lifted from ReForge, original implementation from Comfy
## CFG++ attempt by me

import torch
from tqdm.auto import trange


#   copied from kdiffusion/sampling.py
def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


@torch.no_grad()
def sample_gradient_e(model, x, sigmas, extra_args=None, callback=None, disable=None, ge_gamma=2.):
    """Gradient-estimation sampler. Paper: https://openreview.net/pdf?id=o2ND9v0CeK"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_d = None

    sigmas = sigmas.to(x.device)

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        d = to_d(x, sigmas[i], denoised)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        dt = sigmas[i + 1] - sigmas[i]
        if i == 0: # Euler method
            x = x + d * dt
        else:
            # Gradient estimation
            d_bar = ge_gamma * d + (1 - ge_gamma) * old_d
            x = x + d_bar * dt
        old_d = d
    return x


@torch.no_grad()
def sample_gradient_e_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, ge_gamma=2.):
    """Gradient-estimation sampler. Paper: https://openreview.net/pdf?id=o2ND9v0CeK"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_d = None
    
    model.need_last_noise_uncond = True

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        d = model.last_noise_uncond

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        dt = sigmas[i + 1] - sigmas[i]
        if i == 0: # Euler method
            x = denoised + d * sigmas[i+1]
        else:
            # Gradient estimation
            d_bar = ge_gamma * d + (1 - ge_gamma) * old_d
            x = denoised + d_bar * sigmas[i+1]
        old_d = d
    return x
