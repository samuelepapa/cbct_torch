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
    # Flatten to avoid shape mismatch warnings
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    mse = F.mse_loss(pred_flat, gt_flat)
    if mse == 0:
        return torch.tensor(float("inf"))

    max_val = gt_flat.max()
    return 10 * torch.log10(max_val**2 / mse)
