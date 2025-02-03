# based on -
# Simple Karras-Exponential Scheduler, by Kittensx
# https://github.com/Kittensx/Simple_KES

import torch

import gradio as gr
from k_diffusion.sampling import get_sigmas_karras, get_sigmas_exponential

from modules import shared

def get_sigmas_simple_kes (n, sigma_min, sigma_max, device):
    """
    Scheduler function that blends sigma sequences using Karras and Exponential methods with adaptive parameters.

    Parameters:
        n (int): Number of steps.
        sigma_min (float): Minimum sigma value.
        sigma_max (float): Maximum sigma value.
        device (torch.device): The device on which to perform computations (e.g., 'cuda' or 'cpu').
        start_blend (float): Initial blend factor for dynamic blending.
        end_blend (float): Final blend factor for dynamic blending.
        sharpen_factor (float): Sharpening factor to be applied adaptively.
        update_interval (int): Interval to update blend factors.
        initial_step_size (float): Initial step size for adaptive step size calculation.
        final_step_size (float): Final step size for adaptive step size calculation.
        initial_noise_scale (float): Initial noise scale factor.
        final_noise_scale (float): Final noise scale factor.
        step_size_factor: Adjust to compensate for oversmoothing
        noise_scale_factor: Adjust to provide more variation
        
    Returns:
        torch.Tensor: A tensor of blended sigma values.
    """

    start_blend = getattr(shared.opts, 'kes_start_blend', 0.1)
    end_blend = getattr(shared.opts, 'kes_end_blend', 0.5)
    sharpness = getattr(shared.opts, 'kes_sharpness', 0.95)
    initial_step_size = getattr(shared.opts, 'kes_initial_step_size', 0.9)
    final_step_size = getattr(shared.opts, 'kes_final_step_size', 0.2)
    initial_noise_scale = getattr(shared.opts, 'kes_initial_noise', 1.25)
    final_noise_scale = getattr(shared.opts, 'kes_final_noise', 0.8)
    smooth_blend_factor = getattr(shared.opts, 'kes_smooth_blend', 11)
    step_size_factor = getattr(shared.opts, 'kes_step_size_factor', 0.8)
    noise_scale_factor = getattr(shared.opts, 'kes_noise_scale', 0.9)
    
    # Expand sigma_max slightly to account for smoother transitions
    # sigma_max = sigma_max * 1.1

    # Generate sigma sequences using Karras and Exponential methods
    sigmas_karras = get_sigmas_karras(n=n, sigma_min=sigma_min, sigma_max=sigma_max, device=device)
    sigmas_exponential = get_sigmas_exponential(n=n, sigma_min=sigma_min, sigma_max=sigma_max, device=device)

    # Define progress and initialize blend factor
    progress = torch.linspace(0, 1, len(sigmas_karras)).to(device)
    
    sigs = torch.zeros_like(sigmas_karras).to(device)
    
    # Iterate through each step, dynamically adjust blend factor, step size, and noise scaling
    for i in range(len(sigmas_karras)):
        # Adaptive step size and blend factor calculations
        step_size = initial_step_size * (1 - progress[i]) + final_step_size * progress[i] * step_size_factor  # 0.8 default value Adjusted to avoid over-smoothing

        dynamic_blend_factor = start_blend * (1 - progress[i]) + end_blend * progress[i]

        noise_scale = initial_noise_scale * (1 - progress[i]) + final_noise_scale * progress[i] * noise_scale_factor  # 0.9 default value Adjusted to keep more variation

        # Calculate smooth blending between the two sigma sequences
        smooth_blend = torch.sigmoid((dynamic_blend_factor - 0.5) * smooth_blend_factor) # Increase scaling factor to smooth transitions more
        
        # Compute blended sigma values
        blended_sigma = sigmas_karras[i] * (1 - smooth_blend) + sigmas_exponential[i] * smooth_blend
        
        # Apply step size and noise scaling
        sigs[i] = blended_sigma * step_size * noise_scale

    # Optional: Adaptive sharpening based on sigma values
    sharpen_mask = torch.where(sigs < sigma_min * 1.5, sharpness, 1.0).to(device)
    sigs = sigs * sharpen_mask
    
    if torch.isnan(sigs).any() or torch.isinf(sigs).any():
        raise ValueError("Invalid sigma values detected (NaN or Inf).")

    return sigs.to(device)

shared.options_templates.update(shared.options_section(('simple_kes', "Simple KES", ""), {
    "kes_start_blend":       shared.OptionInfo(0.1,  "start blend factor",       gr.Slider, {"minimum": 0.0,  "maximum": 1.0,  "step": 0.01}),
    "kes_end_blend":         shared.OptionInfo(0.5,  "end blend factor",         gr.Slider, {"minimum": 0.0,  "maximum": 1.0,  "step": 0.01}),
    "kes_sharpness":         shared.OptionInfo(0.95, "sharpness",                gr.Slider, {"minimum": 0.0,  "maximum": 2.0,  "step": 0.01}),
    "kes_initial_step_size": shared.OptionInfo(0.9,  "initial step size",        gr.Slider, {"minimum": 0.01, "maximum": 1.0,  "step": 0.01}), # larger max?
    "kes_final_step_size":   shared.OptionInfo(0.2,  "final step size",          gr.Slider, {"minimum": 0.01, "maximum": 1.0,  "step": 0.01}), #larger max?
    "kes_initial_noise":     shared.OptionInfo(1.25, "initial noise",            gr.Slider, {"minimum": 0.0,  "maximum": 4.0,  "step": 0.01}),
    "kes_final_noise":       shared.OptionInfo(0.8,  "final noise",              gr.Slider, {"minimum": 0.0,  "maximum": 4.0,  "step": 0.01}),
    "kes_smooth_blend":      shared.OptionInfo(11,   "smooth blend factor",      gr.Slider, {"minimum": 0.0,  "maximum": 50.0, "step": 0.1}),
    "kes_step_size_factor":  shared.OptionInfo(0.8,  "step size factor",         gr.Slider, {"minimum": 0.0,  "maximum": 4.0,  "step": 0.01}),
    "kes_noise_scale":       shared.OptionInfo(0.9,  "noise scale factor",       gr.Slider, {"minimum": 0.0,  "maximum": 4.0,  "step": 0.01}),
}))

