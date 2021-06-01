import torch

def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Pass x through unmodified in the forward pass, but scale the gradients
    by `scale` (in [0, 1]) in the backward pass."""
    assert scale <= 1
    assert scale >= 0
    return ((1 - scale) * x).detach() + scale * x

def explained_variance(prediction, target):
    """Compute the percent of total variance that is explained by the model.

    This can be negative if the model is worse than the mean - the model is
    actually increasing the variance of the data.
    """
    sstotal = (target - target.mean()).pow(2).sum()
    residual = target - prediction
    ssresidual = residual.pow(2).sum()
    return 1 - (ssresidual / sstotal)
