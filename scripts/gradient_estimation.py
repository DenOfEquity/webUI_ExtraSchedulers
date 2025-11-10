## lifted from ReForge, original implementation from Comfy
## CFG++ attempt by me

import torch
from tqdm.auto import trange

from k_diffusion.sampling import to_d


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
            x.addcmul_(d, dt)
        else:
            # Gradient estimation
            d_bar = ge_gamma * (d - old_d) + old_d
            x.addcmul_(d_bar, dt)
        old_d = d
    return x


@torch.no_grad()
def sample_gradient_e_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, ge_gamma=2.):
    """Gradient-estimation sampler. Paper: https://openreview.net/pdf?id=o2ND9v0CeK"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_d = None
    
    model.need_last_noise_uncond = True
    model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        
        d = model.last_noise_uncond

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if i == 0: # Euler method
            x = torch.addcmul(denoised, d, sigmas[i+1])
        else:
            # Gradient estimation
            d_bar = ge_gamma * (d - old_d) + old_d
            x = torch.addcmul(denoised, d_bar, sigmas[i+1])
        old_d = d
    return x


@torch.no_grad()
def sample_gradient_e_2s_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., ge_gamma=2.):
    """Gradient-estimation sampler. Paper: https://openreview.net/pdf?id=o2ND9v0CeK"""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_d = None

    model.need_last_noise_uncond = True
    model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_mid = 0.5 * (sigmas[i] + sigmas[i+1])

        d = model.last_noise_uncond

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        if i == 0: # Euler method
            x = denoised + d * sigmas[i+1]
        else:
            # Gradient estimation
            d_bar = ge_gamma * d + (1 - ge_gamma) * old_d
            x_2 = denoised + d_bar * sigma_mid
            old_d = d
            denoised_2 = model(x_2, sigma_mid * s_in, **extra_args)
            d = model.last_noise_uncond
            d_bar = ge_gamma * d + (1 - ge_gamma) * old_d
            x = denoised_2 + d * sigmas[i+1]

        old_d = d

    return x
