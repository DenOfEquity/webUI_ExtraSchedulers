## Extra Schedulers extension for Stable Diffusion webUI ##
### Automatic1111, new Forge (gradio 4 based), and probably reForge ###
#### (webUI must have split sampler/scheduler selection) ####

>[!IMPORTANT]
>not for old Forge. For some related stuff in old Forge, see my old [OverrideScheduler extension](https://github.com/DenOfEquity/SchedRide).

### What do? ###
Adds six new schedulers to the dropdown list:
* cosine: follows a, you guessed it, cosine curve. Initial drop is relatively slow.
* cosine-exponential blend: starts cosine, ends up exponential (long tail).
* phi: (based on original by [Extraltodeus](https://github.com/Extraltodeus/sigmas_tools_and_the_golden_scheduler))
* Laplace: (credit Tiankai et al. (2024), via Comfy)
* Karras Dynamic: (via yoinked-h)
* custom: either a list of sigmas [1.0, 0.6, 0.25, 0.1, 0.0] or an expression that will be evaluated for each sampling step. A list will be log-linear interpolated to the number of sampling steps. A list starting with 1.0 and ending with 0.0 will be scaled between sigma_max and sigma_min. Otherwise list will be interpreted as is.
  * *m*: minimum sigma (adjustable in **Settings**, usually ~0.03)
  * *M*: maximum sigma (adjustable in **Settings**, usually ~14.6)
  * *n*: total steps
  * *s*: this step
  * *x*: step / (total steps - 1)
  * *phi*: (1 + sqrt(5)) / 2

Adds six new samplers:
* Euler a CFG++ [Forge only]
* Euler CFG++ [Forge only]
* Euler Dy CFG++ (based on Euler Dy by Koishi-Star) [Forge only]
* Euler SMEA Dy CFG++ (...) [Forge only]
* Refined Exponential Solver (credit: Katherine Crowson, Birch-san, Clybius)
* DPM++ 4M SDE (credit: Clybius)
  
### Why do? ###
Different results, sometimes better. I tend to use cosine-exponential blend most of the time.

### How do? ###
*(schedulers)* It's just a calculation of different number sequences travelling from sigma_max to sigma_min over the set number of sampling steps, guiding the denoising process. Infinite possibilities, but few sweet spots.

### Redo? ###
Yes, custom scheduler is saved to image infotext and *params.txt*.

### How install? ###
Go to the **Extensions** tab, then **Install from URL**, use the URL for this repository.

Then, go back to the **Installed** tab and hit **Apply and restart UI**.

### more? ###
Check the 'neg' branch instead.
