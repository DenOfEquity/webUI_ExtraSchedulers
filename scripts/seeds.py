# SEEDS implementations by chaObserv : https://github.com/comfyanonymous/ComfyUI/pull/7580

import torch
from tqdm.auto import trange
from k_diffusion.sampling import (
    default_noise_sampler,
)


@torch.no_grad()
def sample_seeds_2(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r=0.5):
    '''
    SEEDS-2 - Stochastic Explicit Exponential Derivative-free Solvers (VE Data Prediction) stage 2
    Arxiv: https://arxiv.org/abs/2305.14267
    '''

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas.to(x.device)

    inject_noise = eta > 0 and s_noise > 0

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, t_next = -sigmas[i].log(), -sigmas[i + 1].log()
            h = t_next - t
            h_eta = h * (eta + 1)
            s = t + r * h
            fac = 1 / (2 * r)
            sigma_s = s.neg().exp()

            coeff_1, coeff_2 = (-r * h_eta).expm1(), (-h_eta).expm1()
            if inject_noise:
                noise_coeff_1 = (-2 * r * h * eta).expm1().neg().sqrt()
                noise_coeff_2 = ((-2 * r * h * eta).expm1() - (-2 * h * eta).expm1()).sqrt()
                noise_1, noise_2 = noise_sampler(sigmas[i], sigma_s), noise_sampler(sigma_s, sigmas[i + 1])

            # Step 1
            x_2 = (coeff_1 + 1) * x - coeff_1 * denoised
            if inject_noise:
                x_2 = x_2 + sigma_s * (noise_coeff_1 * noise_1) * s_noise
            denoised_2 = model(x_2, sigma_s * s_in, **extra_args)

            # Step 2
            denoised_d = (1 - fac) * denoised + fac * denoised_2
            x = (coeff_2 + 1) * x - coeff_2 * denoised_d
            if inject_noise:
                x = x + sigmas[i + 1] * (noise_coeff_2 * noise_1 + noise_coeff_1 * noise_2) * s_noise
    return x

@torch.no_grad()
def sample_seeds_3(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None, r_1=1./3, r_2=2./3):
    '''
    SEEDS-3 - Stochastic Explicit Exponential Derivative-free Solvers (VE Data Prediction) stage 3
    Arxiv: https://arxiv.org/abs/2305.14267
    '''

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas.to(x.device)

    inject_noise = eta > 0 and s_noise > 0

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, t_next = -sigmas[i].log(), -sigmas[i + 1].log()
            h = t_next - t
            h_eta = h * (eta + 1)
            s_1 = t + r_1 * h
            s_2 = t + r_2 * h
            sigma_s_1, sigma_s_2 = s_1.neg().exp(), s_2.neg().exp()

            coeff_1, coeff_2, coeff_3 = (-r_1 * h_eta).expm1(), (-r_2 * h_eta).expm1(), (-h_eta).expm1()
            if inject_noise:
                noise_coeff_1 = (-2 * r_1 * h * eta).expm1().neg().sqrt()
                noise_coeff_2 = ((-2 * r_1 * h * eta).expm1() - (-2 * r_2 * h * eta).expm1()).sqrt()
                noise_coeff_3 = ((-2 * r_2 * h * eta).expm1() - (-2 * h * eta).expm1()).sqrt()
                noise_1, noise_2, noise_3 = noise_sampler(sigmas[i], sigma_s_1), noise_sampler(sigma_s_1, sigma_s_2), noise_sampler(sigma_s_2, sigmas[i + 1])

            # Step 1
            x_2 = (coeff_1 + 1) * x - coeff_1 * denoised
            if inject_noise:
                x_2 = x_2 + sigma_s_1 * (noise_coeff_1 * noise_1) * s_noise
            denoised_2 = model(x_2, sigma_s_1 * s_in, **extra_args)

            # Step 2
            x_3 = (coeff_2 + 1) * x - coeff_2 * denoised + (r_2 / r_1) * (coeff_2 / (r_2 * h_eta) + 1) * (denoised_2 - denoised)
            if inject_noise:
                x_3 = x_3 + sigma_s_2 * (noise_coeff_2 * noise_1 + noise_coeff_1 * noise_2) * s_noise
            denoised_3 = model(x_3, sigma_s_2 * s_in, **extra_args)

            # Step 3
            x = (coeff_3 + 1) * x - coeff_3 * denoised + (1. / r_2) * (coeff_3 / h_eta + 1) * (denoised_3 - denoised)
            if inject_noise:
                x = x + sigmas[i + 1] * (noise_coeff_3 * noise_1 + noise_coeff_2 * noise_2 + noise_coeff_1 * noise_3) * s_noise
    return x
