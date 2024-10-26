import gradio
import math, numpy
import torch
from modules import scripts

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



from scripts.res_solver import sample_res_solver
from modules import sd_samplers_common, sd_samplers
from modules.sd_samplers_kdiffusion import sampler_extra_params, KDiffusionSampler

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
        VPScheduler = schedulers.Scheduler("vp", "VP", get_sigmas_vp)
        LaplaceScheduler = schedulers.Scheduler("laplace", "Laplace", get_sigmas_laplace)

        SineScheduler = schedulers.Scheduler("laplace", "Sine scaled", get_sigmas_sinusoidal_sf)
        InvCosScheduler = schedulers.Scheduler("laplace", "Inverse Cosine scaled", get_sigmas_invcosinusoidal_sf)
        CosDynScheduler = schedulers.Scheduler("laplace", "Cosine Dynamic", get_sigmas_react_cosinusoidal_dynsf)
        KarrasDynScheduler = schedulers.Scheduler("laplace", "Karras Dynamic", get_sigmas_karras_dynamic)
        KarrasExpDecayScheduler = schedulers.Scheduler("laplace", "Karras Exp Decay", get_sigmas_karras_exponential_decay)
        KarrasExpIncScheduler = schedulers.Scheduler("laplace", "Karras Exp Inc", get_sigmas_karras_exponential_increment)

        CustomScheduler = schedulers.Scheduler("custom", "custom", custom_scheduler)


        schedulers.schedulers.append(CosineScheduler)
        schedulers.schedulers.append(CosExpScheduler)
        schedulers.schedulers.append(PhiScheduler)
        schedulers.schedulers.append(VPScheduler)
        schedulers.schedulers.append(LaplaceScheduler)

        schedulers.schedulers.append(SineScheduler)
        schedulers.schedulers.append(InvCosScheduler)
        schedulers.schedulers.append(CosDynScheduler)
        schedulers.schedulers.append(KarrasDynScheduler)
        schedulers.schedulers.append(KarrasExpDecayScheduler)
        schedulers.schedulers.append(KarrasExpIncScheduler)

        schedulers.schedulers.append(CustomScheduler)
        schedulers.schedulers_map = {**{x.name: x for x in schedulers.schedulers}, **{x.label: x for x in schedulers.schedulers}}

        try:
            # CFG++ method is Forge only, not worjing in A1111
            import modules_forge.forge_version
            from scripts.samplers_cfgpp import sample_euler_ancestral_cfgpp, sample_euler_cfgpp, sample_euler_dy_cfgpp, sample_euler_smea_dy_cfgpp, sample_euler_negative_cfgpp, sample_euler_negative_dy_cfgpp
            samplers_cfgpp = [
                ("Euler a CFG++",       sample_euler_ancestral_cfgpp,       ["k_euler_a_cfgpp"],       {"uses_ensd": True}),
                ("Euler CFG++",         sample_euler_cfgpp,                 ["k_euler_cfgpp"],         {}),
                ("Euler Dy CFG++",      sample_euler_dy_cfgpp,              ["k_euler_dy_cfgpp"],      {}),
                ("Euler SMEA Dy CFG++", sample_euler_smea_dy_cfgpp,         ["k_euler_smea_dy_cfgpp"], {}),
                ("Euler Negative CFG++", sample_euler_negative_cfgpp,       ["k_euler_negative_cfgpp"], {}),
                ("Euler Negative Dy CFG++", sample_euler_negative_dy_cfgpp, ["k_euler_negative_dy_cfgpp"], {}),
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

            sd_samplers.all_samplers.extend(samplers_data_cfgpp)
        except:
            pass

        samplers_extra = [
            ("Refined Exponential Solver",           sample_res_solver,                     ["k_res"],              {}),
        ]
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
