# by Clybius : github.com/Clybius/ComfyUI-Extra-Samplers/

import torch
from tqdm.auto import trange


#   copied from kdiffusion/sampling.py and utils.py
def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)


@torch.no_grad()
def sample_clyb_4m_sde_momentumized(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1.0, s_noise=1., noise_sampler=None, momentum=0.0):
    """DPM-Solver++(3M) SDE, modified with an extra SDE, and momentumized in both the SDE and ODE(?). 'its a first' - Clybius 2023
    The expression for d1 is derived from the extrapolation formula given in the paper “Diffusion Monte Carlo with stochastic Hamiltonians” by M. Foulkes, L. Mitas, R. Needs, and G. Rajagopal. The formula is given as follows:
    d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
    (if this is an incorrect citing, we blame Google's Bard and OpenAI's ChatGPT for this and NOT me :^) )

    where d1_0, d1_1, and d1_2 are defined as follows:
    d1_0 = (denoised - denoised_1) / r2
    d1_1 = (denoised_1 - denoised_2) / r1
    d1_2 = (denoised_2 - denoised_3) / r0

    The variables r0, r1, and r2 are defined as follows:
    r0 = h_3 / h_2
    r1 = h_2 / h
    r2 = h / h_1
    """

    def momentum_func(diff, velocity, timescale=1.0, offset=-momentum / 2.0): # Diff is current diff, vel is previous diff
        if velocity is None:
            momentum_vel = diff
        else:
            momentum_vel = momentum * (timescale + offset) * velocity + (1 - momentum * (timescale + offset)) * diff
        return momentum_vel

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()

    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2, denoised_3 = None, None, None
    h_1, h_2, h_3 = None, None, None
    vel, vel_sde = None, None
    for i in trange(len(sigmas) - 1, disable=disable):
        time = sigmas[i] / sigma_max
        denoised = model(x, sigmas[i] * s_in, **extra_args)

        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)
            x_diff = momentum_func((-h_eta).expm1().neg() * denoised, vel, time)
            vel = x_diff
            x = torch.exp(-h_eta) * x + vel

            if h_3 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                r2 = h_3 / h
                d1_0 = (denoised   - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1_2 = (denoised_2 - denoised_3) / r2
                # d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1) + ((d1_0 - d1_1) * r2 / (r1 + r2) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r1 + r2) * (r0 + r1))
                # d2 = (d1_0 - d1_1) / (r0 + r1) + ((d1_0 - d1_1) * r2 / (r1 + r2) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r1 + r2) * (r0 + r1))

                # r0 = h_3 / h_2
                # r1 = h_2 / h
                # r2 = h / h_1
                # d1_0 = (denoised - denoised_1) / r2
                # d1_1 = (denoised_1 - denoised_2) / r1
                # d1_2 = (denoised_2 - denoised_3) / r0
                d1 = d1_0 + (d1_0 - d1_1) * r2 / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) * r2 / ((r2 + r1) * (r0 + r1))
                d2 = (d1_0 - d1_1) / (r2 + r1) + ((d1_0 - d1_1) * r2 / (r2 + r1) - (d1_1 - d1_2) * r1 / (r0 + r1)) / ((r2 + r1) * (r0 + r1))
                phi_3 = h_eta.neg().expm1() / h_eta + 1
                phi_4 = phi_3 / h_eta - 0.5
                sde_diff = momentum_func(phi_3 * d1 - phi_4 * d2, vel_sde, time)
                vel_sde = sde_diff
                x = x + vel_sde
            elif h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                sde_diff = momentum_func(phi_2 * d1 - phi_3 * d2, vel_sde, time)
                vel_sde = sde_diff
                x = x + vel_sde
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                sde_diff = momentum_func(phi_2 * d, vel_sde, time)
                vel_sde = sde_diff
                x = x + vel_sde

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

            denoised_1, denoised_2, denoised_3 = denoised, denoised_1, denoised_2
            h_1, h_2, h_3 = h, h_1, h_2

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

    return x
