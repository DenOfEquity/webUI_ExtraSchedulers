# first 3 lifted from ForgeClassic (https://github.com/Haoming02/sd-webui-forge-classic/)
# 4th is simple adaptation of 3M to 2M
# 5th lifted from ReForge (https://github.com/Panchovix/stable-diffusion-webui-reForge)
# all modified to work with Forge2

import torch
from tqdm.auto import trange
from k_diffusion.sampling import (
    default_noise_sampler,
    BrownianTreeNoiseSampler,
    get_ancestral_step,
    to_d,
)


def _sigma_fn(t):
    return t.neg().exp()


def _t_fn(sigma):
    return sigma.log().neg()


@torch.no_grad()
def sample_dpmpp_sde_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None):
    eta = 1.0
    s_noise = 1.0
    r = 0.5

    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args

    model.need_last_noise_uncond = True
    model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True

    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )

        if sigmas[i + 1] == 0:
            d = model.last_noise_uncond
            x = denoised + d * sigmas[i + 1]
        else:
            t, t_next = _t_fn(sigmas[i]), _t_fn(sigmas[i + 1])
            h = t_next - t
            s = t + h * r
            fac = 1 / (2 * r)


            sd, su = get_ancestral_step(_sigma_fn(t), _sigma_fn(s), eta)
            s_ = _t_fn(sd)
            x_2 = (_sigma_fn(s_) / _sigma_fn(t)) * x - (t - s_).expm1() * denoised
            x_2 = x_2 + noise_sampler(_sigma_fn(t), _sigma_fn(s)) * s_noise * su
            denoised_2 = model(x_2, _sigma_fn(s) * s_in, **extra_args)
            u = x_2 - model.last_noise_uncond * _sigma_fn(s) * s_in                                             #d=(x-u)/sigma; d*sigma=x-u; u=x-d*sigma

            sd, su = get_ancestral_step(_sigma_fn(t), _sigma_fn(t_next), eta)
            denoised_d = (1 - fac) * u + fac * u
            x = denoised_2 + to_d(x, sigmas[i], denoised_d) * sd
            x = x + noise_sampler(_sigma_fn(t), _sigma_fn(t_next)) * s_noise * su
    return x


@torch.no_grad()
def sample_dpmpp_2m_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_uncond_denoised = None
    uncond_denoised = None

    model.need_last_noise_uncond = True
    model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        uncond_denoised = x - model.last_noise_uncond * sigmas[i] * s_in
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        t, t_next = _t_fn(sigmas[i]), _t_fn(sigmas[i + 1])
        h = t_next - t
        if old_uncond_denoised is None or sigmas[i + 1] == 0:
            denoised_mix = -torch.exp(-h) * uncond_denoised
        else:
            h_last = t - _t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_mix = -torch.exp(-h) * uncond_denoised - torch.expm1(-h) * (1 / (2 * r)) * (denoised - old_uncond_denoised)
        x = denoised + denoised_mix + torch.exp(-h) * x
        old_uncond_denoised = uncond_denoised
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=None, s_noise=None, noise_sampler=None):
    eta = 1.0 if eta is None else eta
    s_noise = 1.0 if s_noise is None else s_noise

    if len(sigmas) <= 1:
        return x

    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1, denoised_2 = None, None
    h, h_1, h_2 = None, None, None

    model.need_last_noise_uncond = True
    model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        u = x - model.last_noise_uncond * sigmas[i] * s_in                                            #d=(x-u)/sigma; d*sigma=x-u; u=x-d*sigma
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            h_eta = h * (eta + 1)

            x = torch.exp(-h_eta) * (x + (denoised - u)) + (-h_eta).expm1().neg() * denoised

            if h_2 is not None:
                r0 = h_1 / h
                r1 = h_2 / h
                d1_0 = (denoised - denoised_1) / r0
                d1_1 = (denoised_1 - denoised_2) / r1
                d1 = d1_0 + (d1_0 - d1_1) * r0 / (r0 + r1)
                d2 = (d1_0 - d1_1) / (r0 + r1)
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                phi_3 = phi_2 / h_eta - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x


## extra
@torch.no_grad()
def sample_dpmpp_2m_sde_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
# just cut down from 3m_sde version
    seed = extra_args.get("seed", None)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    denoised_1 = None
    h_1 = None

    model.need_last_noise_uncond = True
    model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        u = x - model.last_noise_uncond * sigmas[i] * s_in
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            #Denoising step
            x = denoised
        else:
            #DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t

            h_eta = h * (eta + 1)
            x = torch.exp(-h_eta) * (x + (denoised - u)) + (-h_eta).expm1().neg() * denoised

            if denoised_1 is not None:
                r = h_1 / h

                d = (denoised - denoised_1) / r
                phi_2 = h_eta.neg().expm1() / h_eta + 1
                x = x + phi_2 * d

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * h * eta).expm1().neg().sqrt() * s_noise

            h_1 = h

        denoised_1 = denoised
    return x


# via ReForge
@torch.no_grad()
def sample_dpmpp_2s_ancestral_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    
    model.need_last_noise_uncond = True
    model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True

    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigma_down == 0:
            # Euler method
            d = model.last_noise_uncond 
            dt = sigma_down - sigmas[i]
            x = denoised + d * sigma_down
        else:
            u = x - model.last_noise_uncond * sigmas[i] * s_in

            # DPM-Solver++(2S)
            t, t_next = t_fn(sigmas[i]), t_fn(sigma_down)
            # r = torch.sinh(1 + (2 - eta) * (t_next - t) / (t - t_fn(sigma_up))) works only on non-cfgpp, weird
            r = 1 / 2
            h = t_next - t
            s = t + r * h
            x_2 = (sigma_fn(s) / sigma_fn(t)) * (x + (denoised - u)) - (-h * r).expm1() * denoised
            denoised_2 = model(x_2, sigma_fn(s) * s_in, **extra_args)
            x = (sigma_fn(t_next) / sigma_fn(t)) * (x + (denoised - u)) - (-h).expm1() * denoised_2

        # Noise addition
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x