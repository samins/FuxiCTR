import torch


def weighted_mse_loss(y_pred, y_true, weight, reduction="mean"):
    """
    Weighed MSE loss
    """
    loss = torch.sum(weight * (y_pred - y_true) ** 2)
    if reduction == "mean":
        loss /= torch.sum(weight)

    return loss
