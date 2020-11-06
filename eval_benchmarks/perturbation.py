import torch.nn as nn
import torch.nn.functional as F
from robustness.tools.helpers import accuracy


class PerturbationBase(nn.Module):
    def __init__(self, args, model, loader):
        super(PerturbationBase, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def _generate_perturbation(self, input):
        raise NotImplementedError("This should be defined on perturbation basis")

    def forward(self, input, targets, logits, model):
        perturbed_input = self._generate_perturbation(input)
        perturbed_logits = model(perturbed_input)
        loss = F.nll_loss(perturbed_logits, targets)
        if perturbed_logits.shape[1] <= 5:
            prec1 = accuracy(perturbed_logits, targets, topk=(1,))
            return {"loss": loss, "top1": prec1[0]}
        else:
            prec1, prec5 = accuracy(perturbed_logits, targets, topk=(1, 5))
            return {"loss": loss, "top1": prec1[0], "top5": prec5[0]}
