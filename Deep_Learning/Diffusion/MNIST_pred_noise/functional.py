import torch
import torch.nn.functional as F

def linear_beta_schedule(timesteps, beta_0, beta_T):
    """
    Linear beta schedule from beta_0 to beta_T
    """
    betas = torch.linspace(beta_0, beta_T, timesteps)
    return betas

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device="cpu"):
    """
    Takes an image and a timestep as input
    and returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def get_loss(model, x_0, t, loss_fn, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    x_noisy, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, model.device)
    noise_pred = model(x_noisy, t)

    assert loss_fn in ['l1', 'l2', 'huber', 'logistic'], "Loss function must be one of 'l1', 'l2', 'huber', 'logistic'"
    if loss_fn == 'l1':
        loss = F.l1_loss(noise_pred, noise)
    elif loss_fn == 'l2':
        loss = F.mse_loss(noise_pred, noise)
    elif loss_fn == 'huber':
        loss = F.smooth_l1_loss(noise_pred, noise)
    elif loss_fn == 'logistic':
        loss = F.binary_cross_entropy_with_logits(noise_pred, noise)
    return loss