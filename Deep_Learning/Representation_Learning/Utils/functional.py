import torch
import torch.nn.functional as F

def NTXent(z:torch.Tensor, temperature:float=0.5):
    """
    Compute the normalized temperature-scaled cross entropy loss for the given batch of samples.
    Args:
        z: torch.Tensor, the batch of samples to compute the loss for.
        temperature: float, the temperature scaling factor.
    Returns:
        torch.Tensor, the computed loss.
    """
    # Compute the cosine similarity matrix
    z = F.normalize(z, dim=-1)
    similarity_matrix = torch.exp(torch.matmul(z, z.T) / temperature)

    # Compute the positive and negative samples
    with torch.no_grad():
        batch_size = z.size(0)
        mask = torch.zeros((batch_size, batch_size), device=z.device, dtype=torch.float32)
        mask[range(1, batch_size, 2), range(0, batch_size, 2)] = 1.0
        mask[range(0, batch_size, 2), range(1, batch_size, 2)] = 1.0
    numerator = similarity_matrix * mask
    denominator = (similarity_matrix * (torch.ones_like(mask) - torch.eye(batch_size, device=z.device))).sum(dim=-1, keepdim=True)

    # prevent nans
    with torch.no_grad():
        numerator[~mask.bool()] = 1.0


    # calculate loss
    losses = -torch.log(numerator / denominator)
    loss = losses[mask.bool()].mean()

    return loss

def smooth_l1_loss(input:torch.Tensor, target:torch.Tensor, beta:float=1.0):
    """
    Compute the smooth L1 loss for the given input and target tensors.
    Args:
        input: torch.Tensor, the input tensor.
        target: torch.Tensor, the target tensor.
        beta: float, the beta parameter.
    Returns:
        torch.Tensor, the computed loss.
    """
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean()