# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

import torch
from . import gaussian_diffusion as gd
from .gaussian_diffusion import rescale_zero_terminal_snr
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    predict_v=False,
    zero_terminal=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    # Rescale to zero-terminal SNR
    if zero_terminal:
        old_betas = torch.from_numpy(betas)
        betas = rescale_zero_terminal_snr(old_betas)
        betas = betas.numpy()

    # Loss Type
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]

    # Mean type
    if predict_xstart:
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_v:
        model_mean_type = gd.ModelMeanType.VELOCITY
    else:
        model_mean_type = gd.ModelMeanType.EPSILON

    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type
        # rescale_timesteps=rescale_timesteps,
    )
