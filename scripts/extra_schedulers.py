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

    if "name=\'custom\'" not in str(schedulers.schedulers[-1]):
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
            ("Euler a CFG++", sample_euler_ancestral_cfgpp, ["k_euler_a_cfgpp"], {"uses_ensd": True}),
            ("Euler CFG++", sample_euler_cfgpp, ["k_euler_cfgpp"], {}),
        ]
        samplers_data_cfgpp = [
            sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
            for label, funcname, aliases, options in samplers_cfgpp
            if callable(funcname)
        ]
        sampler_extra_params['sample_euler_cfgpp'] = ['s_churn', 's_tmin', 's_tmax', 's_noise']

        sd_samplers.all_samplers.extend(samplers_data_cfgpp)
        sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
        sd_samplers.set_samplers()
        
    ExtraScheduler.installed = True
except:
    print ("Extension: Extra Schedulers: unsupported webUI")
    ExtraScheduler.installed = False
