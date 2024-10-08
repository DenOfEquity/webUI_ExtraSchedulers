import gradio
import math, numpy
import torch
from modules import scripts
from tqdm.auto import trange

#   copied from kdiffusion/sampling.py
def default_noise_sampler(x):
    return lambda sigma, sigma_next: torch.randn_like(x)
def get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up

def cosine_scheduler (n, sigma_min, sigma_max, device):
    sigmas = torch.zeros(n, device=device)
    if n == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        for x in range(n):
            p = x / (n-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas[x] = C
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def cosexpblend_scheduler (n, sigma_min, sigma_max, device):
    sigmas = []
    if n == 1:
        sigmas.append(sigma_max ** 0.5)
    else:
        K = (sigma_min / sigma_max)**(1/(n-1))
        E = sigma_max
        for x in range(n):
            p = x / (n-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas.append(C + p * (E - C))
            E *= K
    sigmas += [0.0]
    return torch.FloatTensor(sigmas).to(device)

##  phi scheduler modified from original by @extraltodeus
def phi_scheduler(n, sigma_min, sigma_max, device):
    sigmas = torch.zeros(n, device=device)
    if n == 1:
        sigmas[0] = sigma_max ** 0.5
    else:
        phi = (1 + 5**0.5) / 2
        for x in range(n):
            sigmas[x] = sigma_min + (sigma_max-sigma_min)*((1-x/(n-1))**(phi*phi))
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def custom_scheduler(n, sigma_min, sigma_max, device):
    if 'import' in ExtraScheduler.customSigmas:
        sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
    elif 'eval' in ExtraScheduler.customSigmas:
        sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
    elif 'scripts' in ExtraScheduler.customSigmas:
        sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)

    elif ExtraScheduler.customSigmas[0] == '[' and ExtraScheduler.customSigmas[-1] == ']':
        sigmasList = [float(x) for x in ExtraScheduler.customSigmas.strip('[]').split(',')]

        if sigmasList[0] == 1.0 and sigmasList[-1] == 0.0:
            for x in range(len(sigmasList)):
                sigmasList[x] *= (sigma_max - sigma_min)
                sigmasList[x] += sigma_min

        xs = numpy.linspace(0, 1, len(sigmasList))
        ys = numpy.log(sigmasList[::-1])
        
        new_xs = numpy.linspace(0, 1, n)
        new_ys = numpy.interp(new_xs, xs, ys)
        
        interpolated_ys = numpy.exp(new_ys)[::-1].copy()
        sigmas = torch.tensor(interpolated_ys, device=device)
    else:
        sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
        
        phi = (1 + 5**0.5) / 2
        pi = math.pi
        
        s = 0
        while (s < n):
            x = (s) / (n - 1)
            M = sigma_max
            m = sigma_min
        
            sigmas[s] = eval((ExtraScheduler.customSigmas))
            s += 1
    return torch.cat([sigmas, sigmas.new_zeros([1])])



@torch.no_grad()
def sample_euler_cfgpp(model, x, sigmas, extra_args=None, callback=None, disable=None, s_churn=0., s_tmin=0., s_tmax=float('inf'), s_noise=1.):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    extra_args = {} if extra_args is None else extra_args
    model.need_last_noise_uncond = True
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
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
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
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
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        gamma = min(s_churn / (len(sigmas) - 1), 2 ** 0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat ** 2 - sigmas[i] ** 2) ** 0.5
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

class ExtraScheduler(scripts.Script):
    sorting_priority = 99

    installed = False
    customSigmas = 'm + (M-m)*(1-x)**3'

    def title(self):
        return "Extra Schedulers (custom)"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        if ExtraScheduler.installed:
            return scripts.AlwaysVisible
        else:
            return False

    def ui(self, *args, **kwargs):
        #with gradio.Accordion(open=False, label=self.title(), visible=ExtraScheduler.installed):
        custom_sigmas = gradio.Textbox(value=ExtraScheduler.customSigmas, label='Extra Schedulers: custom function / list [n0, n1, n2, ...]', lines=1.01)

        self.infotext_fields = [
            (custom_sigmas, "es_custom"),
        ]
        return [custom_sigmas]

    def process(self, params, *script_args, **kwargs):
        if params.scheduler == 'custom':
            custom_sigmas = script_args[0]
            ExtraScheduler.customSigmas = custom_sigmas
            params.extra_generation_params.update(dict(es_custom = ExtraScheduler.customSigmas, ))
        return

try:
    import modules.sd_schedulers as schedulers

    if "name='custom'" not in str(schedulers.schedulers[-1]):
        print ("Extension: Extra Schedulers: adding new schedulers")
        CosineScheduler = schedulers.Scheduler("cosine", "Cosine", cosine_scheduler)
        CosExpScheduler = schedulers.Scheduler("cosexp", "CosineExponential blend", cosexpblend_scheduler)
        PhiScheduler = schedulers.Scheduler("phi", "Phi", phi_scheduler)
        CustomScheduler = schedulers.Scheduler("custom", "custom", custom_scheduler)
        schedulers.schedulers.append(CosineScheduler)
        schedulers.schedulers.append(CosExpScheduler)
        schedulers.schedulers.append(PhiScheduler)
        schedulers.schedulers.append(CustomScheduler)
        schedulers.schedulers_map = {**{x.name: x for x in schedulers.schedulers}, **{x.label: x for x in schedulers.schedulers}}
        
        from modules import sd_samplers_common, sd_samplers
        from modules.sd_samplers_kdiffusion import sampler_extra_params, KDiffusionSampler
        samplers_cfgpp = [
            ("Euler a CFG++",       sample_euler_ancestral_cfgpp, ["k_euler_a_cfgpp"],       {"uses_ensd": True}),
            ("Euler CFG++",         sample_euler_cfgpp,           ["k_euler_cfgpp"],         {}),
            ("Euler Dy CFG++",      sample_euler_dy_cfgpp,        ["k_euler_dy_cfgpp"],      {}),
            ("Euler SMEA Dy CFG++", sample_euler_smea_dy_cfgpp,   ["k_euler_smea_dy_cfgpp"], {}),
        ]
        samplers_data_cfgpp = [
            sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
            for label, funcname, aliases, options in samplers_cfgpp
            if callable(funcname)
        ]
        sampler_extra_params['sample_euler_cfgpp']         = ['s_churn', 's_tmin', 's_tmax', 's_noise']
        sampler_extra_params['sample_euler_dy_cfgpp']      = ['s_churn', 's_tmin', 's_tmax', 's_noise']
        sampler_extra_params['sample_euler_smea_dy_cfgpp'] = ['s_churn', 's_tmin', 's_tmax', 's_noise']

        sd_samplers.all_samplers.extend(samplers_data_cfgpp)
        sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
        sd_samplers.set_samplers()
        
    ExtraScheduler.installed = True
except:
    print ("Extension: Extra Schedulers: unsupported webUI")
    ExtraScheduler.installed = False
