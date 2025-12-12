import torch
import torch.nn.functional as F


def psnr(pred, gt):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).

    Args:
        pred (torch.Tensor): Predicted image.
        gt (torch.Tensor): Ground truth image.

    Returns:
        torch.Tensor: PSNR value.
    """
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return torch.tensor(float("inf"))

    max_val = gt.max()
    return 10 * torch.log10(max_val**2 / mse)
