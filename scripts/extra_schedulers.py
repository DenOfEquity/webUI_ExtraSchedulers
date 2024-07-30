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
        custom_sigmas = script_args[0]

        ExtraScheduler.customSigmas = custom_sigmas
        params.extra_generation_params.update(dict(es_custom = ExtraScheduler.customSigmas, ))
        return

try:
    import modules.sd_schedulers as schedulers

    if "name=\'custom\'" in str(schedulers.schedulers[-1]):
        print ("Extension: Extra Schedulers: removing schedulers")
        del schedulers.schedulers[-4:]

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
    ExtraScheduler.installed = True
except:
    print ("Extension: Extra Schedulers: unsupported webUI")

