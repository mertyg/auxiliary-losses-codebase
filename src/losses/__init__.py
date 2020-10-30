import torch.nn as nn
from .gaussian_aug import GaussianAug

__losses__ = {"gaussian-kl": GaussianAug}


def get_loss(args, model, loader) -> nn.Module:
    loss_args = args.custom_loss.split("_")
    loss_type = loss_args[0]
    if loss_type not in __losses__:
        raise NotImplementedError(f"Loss={loss_type} is not implemented yet.")
    method = __losses__[loss_type]
    return method(args, model, loader)
