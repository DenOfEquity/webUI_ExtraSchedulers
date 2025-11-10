import gradio
import math
import numpy
import torch
from modules import scripts, shared


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


def cosexpblend_boost_scheduler (n, sigma_min, sigma_max, device):
    sigmas = []
    if n == 1:
        sigmas.append(sigma_max ** 0.5)
    else:
        detail = numpy.interp(numpy.linspace(0, 1, n), numpy.linspace(0, 1, 5), [1.0, 1.0, 1.27, 1.0, 1.0])

        K = (sigma_min / sigma_max)**(1/(n-1))
        E = sigma_max
        for x in range(n):
            p = x / (n-1)
            C = sigma_min + 0.5*(sigma_max-sigma_min)*(1 - math.cos(math.pi*(1 - p**0.5)))
            sigmas.append(detail[x] * (C + p * (E - C)))
            E *= K

    sigmas += [0.0]

    return torch.FloatTensor(sigmas).to(device)


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


def get_sigmas_vp(n, sigma_min, sigma_max, device='cpu'):
    """Constructs a continuous VP noise schedule."""
    
    beta_d = 19.9
    beta_min = 0.1
    eps_s = 1e-3
    
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_laplace(n, sigma_min, sigma_max, device='cpu'):
    """Constructs the noise schedule proposed by Tiankai et al. (2024). """
    mu = 0.
    beta = 0.5
    epsilon = 1e-5 # avoid log(0)
    x = torch.linspace(0, 1, n, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)
    lmb = mu - beta * torch.sign(0.5-x) * torch.log(1 - 2 * torch.abs(0.5-x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_sinusoidal_sf(n, sigma_min, sigma_max, device='cpu'):
    """Constructs a sinusoidal noise schedule."""
    sf = 3.5
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (1 - torch.sin(torch.pi / 2 * x)))/sigma_max
    sigmas = sigmas**sf
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])
 
 
def get_sigmas_invcosinusoidal_sf(n, sigma_min, sigma_max, device='cpu'):
    """Constructs a sinusoidal noise schedule."""
    sf = 3.5
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min + (sigma_max - sigma_min) * (0.5*(torch.cos(x * math.pi) + 1)))/sigma_max
    sigmas = sigmas**sf
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])
 
 
def get_sigmas_react_cosinusoidal_dynsf(n, sigma_min, sigma_max, device='cpu'):
    """Constructs a sinusoidal noise schedule."""
    sf = 2.15
    x = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_min+(sigma_max-sigma_min)*(torch.cos(x*(torch.pi/2))))/sigma_max
    sigmas = sigmas**(sf*(n*x/n))
    sigmas = sigmas * sigma_max
    return torch.cat([sigmas, sigmas.new_zeros([1])])
 
 
def get_sigmas_karras_dynamic(n, sigma_min, sigma_max, device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    rho = 7.
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = torch.zeros_like(ramp)
    for i in range(n):
        sigmas[i] = (max_inv_rho + ramp[i] * (min_inv_rho - max_inv_rho)) ** (math.cos(i*math.tau/n)*2+rho) 
    return torch.cat([sigmas, sigmas.new_zeros([1])])
 
 
def get_sigmas_karras_exponential_decay(n, sigma_min, sigma_max, device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    rho = 7.
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = torch.zeros_like(ramp)
    for i in range(n):
        sigmas[i] = (max_inv_rho + ramp[i] * (min_inv_rho - max_inv_rho)) ** (rho-(3*i/n))
    return torch.cat([sigmas, sigmas.new_zeros([1])])
 
 
def get_sigmas_karras_exponential_increment(n, sigma_min, sigma_max, device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    rho = 7.
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = torch.zeros_like(ramp)
    for i in range(n):
        sigmas[i] = (max_inv_rho + ramp[i] * (min_inv_rho - max_inv_rho)) ** (rho+3*i/n)
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
        elif sigmasList[-1] == 0.0:
            #don't interpolate to number of steps, use as is
            return torch.tensor(sigmasList)

        xs = numpy.linspace(0, 1, len(sigmasList))
        ys = numpy.log(sigmasList[::-1])
        
        new_xs = numpy.linspace(0, 1, n)
        new_ys = numpy.interp(new_xs, xs, ys)
        
        interpolated_ys = numpy.exp(new_ys)[::-1].copy()
        sigmas = torch.tensor(interpolated_ys, device=device)
    else:
        sigmas = torch.linspace(sigma_max, sigma_min, n, device=device)
        detail = numpy.interp(numpy.linspace(0, 1, n), numpy.linspace(0, 1, 5), [1.0, 1.0, 1.25, 1.0, 1.0])

        phi = (1 + 5**0.5) / 2
        pi = math.pi
        
        s = 0
        while (s < n):
            x = (s) / (n - 1)
            M = sigma_max
            m = sigma_min
            d = detail[s]
        
            sigmas[s] = eval((ExtraScheduler.customSigmas))
            s += 1

    return torch.cat([sigmas, sigmas.new_zeros([1])])


from scripts.simple_kes import get_sigmas_simple_kes

from scripts.res_solver import sample_res_solver, sample_res_multistep, sample_res_multistep_cfgpp
from scripts.clybius_dpmpp_4m_sde import sample_clyb_4m_sde_momentumized
from scripts.gradient_estimation import sample_gradient_e, sample_gradient_e_cfgpp, sample_gradient_e_2s_cfgpp
from scripts.seeds import sample_seeds_2, sample_seeds_3

from modules import sd_samplers_common, sd_samplers
from modules.sd_samplers_kdiffusion import sampler_extra_params, KDiffusionSampler


class ExtraScheduler(scripts.Script):
    sorting_priority = 99

    installed = False
    customSigmas = 'm + (M-m)*(1-x)**3'

    def title(self):
        return "Extra Schedulers (custom)"

    def show(self, is_img2img):
        if ExtraScheduler.installed:
            return scripts.AlwaysVisible
        else:
            return False

    def ui(self, *args, **kwargs):
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
        elif params.scheduler == 'Simple KES':
            params.extra_generation_params.update(dict(
                es_KES_start_blend       = getattr(shared.opts, 'kes_start_blend'),
                es_KES_end_blend         = getattr(shared.opts, 'kes_end_blend'),
                es_KES_sharpness         = getattr(shared.opts, 'kes_sharpness'),
                es_KES_initial_step_size = getattr(shared.opts, 'kes_initial_step_size'),
                es_KES_final_step_size   = getattr(shared.opts, 'kes_final_step_size'),
                es_KES_initial_noise     = getattr(shared.opts, 'kes_initial_noise'),
                es_KES_final_noise       = getattr(shared.opts, 'kes_final_noise'),
                es_KES_smooth_blend      = getattr(shared.opts, 'kes_smooth_blend'),
                es_KES_step_size_factor  = getattr(shared.opts, 'kes_step_size_factor'),
                es_KES_noise_scale       = getattr(shared.opts, 'kes_noise_scale'),
            ))
        return


try:
    import modules.sd_schedulers as schedulers

    if "name='custom'" not in str(schedulers.schedulers[-1]):   # this is a bit lazy tbh
        print ("Extension: Extra Schedulers: adding new schedulers")
        CosineScheduler         = schedulers.Scheduler("cosine",        "Cosine",                   cosine_scheduler)
        CosExpScheduler         = schedulers.Scheduler("cosexp",        "CosineExponential blend",  cosexpblend_scheduler)
        CosExpBScheduler        = schedulers.Scheduler("cosprev",       "CosExp blend boost",       cosexpblend_boost_scheduler)
        PhiScheduler            = schedulers.Scheduler("phi",           "Phi",                      phi_scheduler)
        VPScheduler             = schedulers.Scheduler("vp",            "VP",                       get_sigmas_vp)
        LaplaceScheduler        = schedulers.Scheduler("laplace",       "Laplace",                  get_sigmas_laplace)

        SineScheduler           = schedulers.Scheduler("sine_sc",       "Sine scaled",              get_sigmas_sinusoidal_sf)
        InvCosScheduler         = schedulers.Scheduler("inv_cos_sc",    "Inverse Cosine scaled",    get_sigmas_invcosinusoidal_sf)
        CosDynScheduler         = schedulers.Scheduler("cosine_dyn",    "Cosine Dynamic",           get_sigmas_react_cosinusoidal_dynsf)
        KarrasDynScheduler      = schedulers.Scheduler("karras_dyn",    "Karras Dynamic",           get_sigmas_karras_dynamic)
        KarrasExpDecayScheduler = schedulers.Scheduler("karras_exp_d",  "Karras Exp Decay",         get_sigmas_karras_exponential_decay)
        KarrasExpIncScheduler   = schedulers.Scheduler("karras_exp_i",  "Karras Exp Inc",           get_sigmas_karras_exponential_increment)

        SimpleKEScheduler       = schedulers.Scheduler("simple_kes",    "Simple KES",               get_sigmas_simple_kes)

        CustomScheduler         = schedulers.Scheduler("custom",        "custom",                   custom_scheduler)


        schedulers.schedulers.append(CosineScheduler)
        schedulers.schedulers.append(CosExpScheduler)
        schedulers.schedulers.append(CosExpBScheduler)
        schedulers.schedulers.append(PhiScheduler)
        schedulers.schedulers.append(VPScheduler)
        schedulers.schedulers.append(LaplaceScheduler)

        schedulers.schedulers.append(SineScheduler)
        schedulers.schedulers.append(InvCosScheduler)
        schedulers.schedulers.append(CosDynScheduler)
        schedulers.schedulers.append(KarrasDynScheduler)
        schedulers.schedulers.append(KarrasExpDecayScheduler)
        schedulers.schedulers.append(KarrasExpIncScheduler)

        schedulers.schedulers.append(SimpleKEScheduler)

        schedulers.schedulers.append(CustomScheduler)
        schedulers.schedulers_map = {**{x.name: x for x in schedulers.schedulers}, **{x.label: x for x in schedulers.schedulers}}

        try:
            # CFG++ method is Forge only, not working in A1111
            import modules_forge.forge_version
            from scripts.samplers_cfgpp import sample_euler_ancestral_cfgpp, sample_euler_cfgpp, sample_euler_dy_cfgpp, sample_euler_smea_dy_cfgpp, sample_euler_negative_cfgpp, sample_euler_negative_dy_cfgpp
            from scripts.forgeClassic_cfgpp import sample_dpmpp_sde_cfgpp, sample_dpmpp_2m_cfgpp, sample_dpmpp_2m_sde_cfgpp, sample_dpmpp_3m_sde_cfgpp, sample_dpmpp_2s_ancestral_cfgpp
            samplers_cfgpp = [
                ("Euler a CFG++",           sample_euler_ancestral_cfgpp,   ["k_euler_a_cfgpp"],            {"uses_ensd": True} ),
                ("Euler CFG++",             sample_euler_cfgpp,             ["k_euler_cfgpp"],              {}                  ),
                ("Euler Dy CFG++",          sample_euler_dy_cfgpp,          ["k_euler_dy_cfgpp"],           {}                  ),
                ("Euler SMEA Dy CFG++",     sample_euler_smea_dy_cfgpp,     ["k_euler_smea_dy_cfgpp"],      {}                  ),
                ("Euler Negative CFG++",    sample_euler_negative_cfgpp,    ["k_euler_negative_cfgpp"],     {}                  ),
                ("Euler Negative Dy CFG++", sample_euler_negative_dy_cfgpp, ["k_euler_negative_dy_cfgpp"],  {}                  ),
                ("RES multistep CFG++",     sample_res_multistep_cfgpp,     ["k_res_multi_cfgpp"],          {}                  ),
                ("Gradient Estimation CFG++", sample_gradient_e_cfgpp,      ["k_grad_est_cfgpp"],           {}                  ),
                ("Gradient Estimation 2S CFG++", sample_gradient_e_2s_cfgpp,["k_ge2s_cfgpp"],               {"second_order": True} ),
                ("DPM++ SDE CFG++",         sample_dpmpp_sde_cfgpp,         ["k_dpmpp_sde_cfgpp"],          {"brownian_noise": True, "second_order": True} ),
                ("DPM++ 2M CFG++",          sample_dpmpp_2m_cfgpp,          ["k_dpmpp_2m_cfgpp"],           {}                  ),
                ("DPM++ 2M SDE CFG++",      sample_dpmpp_2m_sde_cfgpp,      ["k_dpmpp_2m_sde_cfgpp"],       {"brownian_noise": True} ),
                ("DPM++ 3M SDE CFG++",      sample_dpmpp_3m_sde_cfgpp,      ["k_dpmpp_3m_sde_cfgpp"],       {"brownian_noise": True, 'discard_next_to_last_sigma': True} ),
                ("DPM++ 2S a CFG++",        sample_dpmpp_2s_ancestral_cfgpp,["k_dpmpp_2s_a_cfgpp"],         {"uses_ensd": True, "second_order": True} ),
            ]
            samplers_data_cfgpp = [
                sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
                for label, funcname, aliases, options in samplers_cfgpp
                if callable(funcname)
            ]
            sampler_extra_params['sample_euler_cfgpp']             = ['s_churn', 's_tmin', 's_tmax', 's_noise']
            sampler_extra_params['sample_euler_negative_cfgpp']    = ['s_churn', 's_tmin', 's_tmax', 's_noise']
            sampler_extra_params['sample_euler_dy_cfgpp']          = ['s_churn', 's_tmin', 's_tmax', 's_noise']
            sampler_extra_params['sample_euler_negative_dy_cfgpp'] = ['s_churn', 's_tmin', 's_tmax', 's_noise']
            sampler_extra_params['sample_euler_smea_dy_cfgpp']     = ['s_churn', 's_tmin', 's_tmax', 's_noise']

            sampler_extra_params['sample_dpmpp_sde_cfgpp']         = ['s_noise']
            sampler_extra_params['sample_dpmpp_2m_sde_cfgpp']      = ['s_noise']
            sampler_extra_params['sample_dpmpp_3m_sde_cfgpp']      = ['s_noise']
            sampler_extra_params['sample_dpmpp_2s_ancestral_cfgpp']= ['s_noise']
            sampler_extra_params['sample_gradient_e_2s_cfgpp']     = ['s_noise']

            sd_samplers.all_samplers.extend(samplers_data_cfgpp)
        except:
            pass

        samplers_extra = [
            ("RES multistep",                sample_res_multistep,              ["k_res_multi"],        {}),
            ("Refined Exponential Solver",   sample_res_solver,                 ["k_res"],              {}),
            ("DPM++ 4M SDE",                 sample_clyb_4m_sde_momentumized,   ["k_dpmpp_4m_sde"],     {}),
            ("Gradient Estimation",          sample_gradient_e,                 ["k_grad_est"],         {}),
            ("SEEDS-2",                      sample_seeds_2,                    ["k_seeds2"],           {}),
            ("SEEDS-3",                      sample_seeds_3,                    ["k_seeds3"],           {}),
        ]

        sampler_extra_params['sample_seeds_2'] = ['s_noise']
        sampler_extra_params['sample_seeds_3'] = ['s_noise']

        samplers_data_extra = [
            sd_samplers_common.SamplerData(label, lambda model, funcname=funcname: KDiffusionSampler(funcname, model), aliases, options)
            for label, funcname, aliases, options in samplers_extra
            if callable(funcname)
        ]

        sd_samplers.all_samplers.extend(samplers_data_extra)
        sd_samplers.all_samplers_map = {x.name: x for x in sd_samplers.all_samplers}
        sd_samplers.set_samplers()

    ExtraScheduler.installed = True
except:
    print ("Extension: Extra Schedulers: unsupported webUI")
    ExtraScheduler.installed = False
