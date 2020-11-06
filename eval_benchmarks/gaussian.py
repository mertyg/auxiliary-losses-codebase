from .perturbation import PerturbationBase
import torch


class Gaussian(PerturbationBase):
    def __init__(self, args, model, loader):
        super(Gaussian, self).__init__(args, model, loader)
        loss_args = args.eval_benchmark.split("_")
        self.std = float(loss_args[1])
        self.args = args

    def _generate_perturbation(self, input):
        z = torch.randn(*input.shape).to(self.args.device)
        return z*self.std + input
