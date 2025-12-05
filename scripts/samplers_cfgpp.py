import torch
from tqdm.auto import trange

from k_diffusion.sampling import (
    default_noise_sampler,
    get_ancestral_step,
)


@torch.no_grad()
def sample_euler_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    model.need_last_noise_uncond = True
    model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True
    s_in = x.new_ones([x.shape[0]])

    if s_churn > 0.0:
        seed = (int(x[0,0,0,0].item()) * 1234567890) % 65536
        generator = torch.Generator(device='cpu').manual_seed(seed)
    else:
        generator = None

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn(x.shape, generator=generator).to(x) * s_noise
            x.add_(eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5)
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = model.last_noise_uncond

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        # Euler method
        x = denoised + d * sigmas[i+1]
    return x


class _Rescaler:
    def __init__(self, model, x, mode, **extra_args):
        self.model = model
        self.x = x
        self.mode = mode
        self.extra_args = extra_args
        self.init_latent, self.mask, self.nmask = model.init_latent, model.mask, model.nmask

    def __enter__(self):
        if self.init_latent is not None:
            self.model.init_latent = torch.nn.functional.interpolate(input=self.init_latent, size=self.x.shape[2:4], mode=self.mode)
        if self.mask is not None:
            self.model.mask = torch.nn.functional.interpolate(input=self.mask.unsqueeze(0), size=self.x.shape[2:4], mode=self.mode).squeeze(0)
        if self.nmask is not None:
            self.model.nmask = torch.nn.functional.interpolate(input=self.nmask.unsqueeze(0), size=self.x.shape[2:4], mode=self.mode).squeeze(0)

        return self

    def __exit__(self, type, value, traceback):
        del self.model.init_latent, self.model.mask, self.model.nmask
        self.model.init_latent, self.model.mask, self.model.nmask = self.init_latent, self.mask, self.nmask

@torch.no_grad()
def dy_sampling_step_cfgpp(x, model, sigma_hat, **extra_args):
    original_shape = x.shape
    batch_size, channels, m, n = original_shape[0], original_shape[1], original_shape[2] // 2, original_shape[3] // 2
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]
    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, channels, m * n, 2, 2)
    c = a_list[:, :, :, 1, 1].view(batch_size, channels, m, n)

    with _Rescaler(model, c, 'nearest-exact', **extra_args) as rescaler:
        denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **rescaler.extra_args)
    d = model.last_noise_uncond
    c = denoised + d * sigma_hat

    d_list = c.view(batch_size, channels, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]
    x = a_list.view(batch_size, channels, m, n, 2, 2).permute(0, 1, 2, 4, 3, 5).reshape(batch_size, channels, 2 * m, 2 * n)

    if extra_row or extra_col:
        x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        x_expanded[:, :, :2 * m, :2 * n] = x
        if extra_row:
            x_expanded[:, :, -1:, :2 * n + 1] = extra_row_content
        if extra_col:
            x_expanded[:, :, :2 * m, -1:] = extra_col_content
        if extra_row and extra_col:
            x_expanded[:, :, -1:, -1:] = extra_col_content[:, :, -1:, :]
        x = x_expanded

    return x

@torch.no_grad()
def smea_sampling_step_cfgpp(x, model, sigma_hat, **extra_args):
    m, n = x.shape[2], x.shape[3]
    x = torch.nn.functional.interpolate(input=x, scale_factor=(1.25, 1.25), mode='nearest-exact')
    with _Rescaler(model, x, 'nearest-exact', **extra_args) as rescaler:
        denoised = model(x, sigma_hat * x.new_ones([x.shape[0]]), **rescaler.extra_args)
    d = model.last_noise_uncond
    x = denoised + d * sigma_hat
    x = torch.nn.functional.interpolate(input=x, size=(m,n), mode='nearest-exact')
    return x


@torch.no_grad()
def sample_euler_dy_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """CFG++ version of Euler Dy by KoishiStar."""
    extra_args = {} if extra_args is None else extra_args
    model.need_last_noise_uncond = True
    model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True
    s_in = x.new_ones([x.shape[0]])

    if s_churn > 0.0:
        seed = (int(x[0,0,0,0].item()) * 1234567890) % 65536
        generator = torch.Generator(device='cpu').manual_seed(seed)
    else:
        generator = None

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn(x.shape, generator=generator).to(x) * s_noise
            x .add_(eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5)
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = model.last_noise_uncond

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        # Euler method
        x = denoised + d * sigmas[i+1]

        if sigmas[i + 1] > 0:
            if i // 2 == 1:
                x = dy_sampling_step_cfgpp(x, model, sigma_hat, **extra_args)        

    return x


@torch.no_grad()
def sample_euler_smea_dy_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """CFG++ version of Euler SMEA Dy by KoishiStar."""
    extra_args = {} if extra_args is None else extra_args
    model.need_last_noise_uncond = True
    model.inner_model.inner_model.forge_objects.unet.model_options["disable_cfg1_optimization"] = True
    s_in = x.new_ones([x.shape[0]])

    if s_churn > 0.0:
        seed = (int(x[0,0,0,0].item()) * 1234567890) % 65536
        generator = torch.Generator(device='cpu').manual_seed(seed)
    else:
        generator = None

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            eps = torch.randn(x.shape, generator=generator).to(x) * s_noise
            x.add_(eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5)
        denoised = model(x, sigma_hat * s_in, **extra_args)
        d = model.last_noise_uncond

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigma_hat, 'denoised': denoised})

        # Euler method
        x = denoised + d * sigmas[i+1]
        
        if sigmas[i + 1] > 0:
            if i + 1 // 2 == 1:     #   ??  this is i == 1; why not if i // 2 == 1 same as Euler Dy
                x = dy_sampling_step_cfgpp(x, model, sigma_hat, **extra_args)
            if i + 1 // 2 == 0:     #   ??  this is i == 0
                x = smea_sampling_step_cfgpp(x, model, sigma_hat, **extra_args)        
    return x


@torch.no_grad()
def sample_euler_ancestral_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, eta=1., s_noise=1., noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    model.need_last_noise_uncond = True
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        d = model.last_noise_uncond

        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

        # Euler method
        x = denoised + d * sigma_down
        if sigmas[i + 1] > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x
