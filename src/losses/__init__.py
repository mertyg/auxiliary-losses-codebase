import torch.nn as nn
from .gaussian_aug import GaussianAug
from .adversarial_attack import AdversarialAttack

__losses__ = {"gaussian-kl": GaussianAug,
              "adversarial-attack": AdversarialAttack}


def get_loss(args, model, loader) -> nn.Module:
    if not args.custom_loss:
        return None
    loss_args = args.custom_loss.split("_")
    loss_type = loss_args[0]
    if loss_type not in __losses__:
        raise NotImplementedError(f"Loss={loss_type} is not implemented yet.")
    method = __losses__[loss_type]
    return method(args, model, loader)
