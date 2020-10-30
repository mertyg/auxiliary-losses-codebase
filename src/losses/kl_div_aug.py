import torch
import torch.nn as nn


class KL_aug_base(nn.Module):
    def __init__(self, args, model, loader):
        super(KL_aug_base, self).__init__()
        self.custom_loss_weight = args.custom_loss_weight
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _sample_noise(self, input):
        raise NotImplementedError("This should be defined on augmentation basis")

    def _kl_div(self, logits, perturbed_logits):
        p = self.softmax(logits)
        log_p = self.log_softmax(logits)
        log_q = self.log_softmax(perturbed_logits)
        kl_loss = torch.sum(p * (log_p - log_q), dim=-1)
        return kl_loss

    def forward(self, input, targets, logits, model):
        """
        Perturbs the input with gaussian noise, then evaluates kl-divergence between predictive distributions.
        """
        noise = self._sample_noise(input)
        perturbed = input + noise
        perturbed_logits = model(perturbed)
        kl_loss = self._kl_div(logits, perturbed_logits)
        kl_loss = self.custom_loss_weight * torch.mean(kl_loss)
        return kl_loss
