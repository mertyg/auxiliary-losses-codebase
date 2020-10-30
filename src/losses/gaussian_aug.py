import torch
import torch.nn as nn
from .kl_div_aug import KL_aug_base


class GaussianAug(KL_aug_base):
    def __init__(self, args, model, loader):
        super(GaussianAug, self).__init__(args, model, loader)
        loss_args = args.custom_loss.split("_")
        self.std = float(loss_args[1])
        self.args = args

    def _sample_noise(self, input):
        z = torch.randn(*input.shape).to(self.args.device)
        return z*self.std
