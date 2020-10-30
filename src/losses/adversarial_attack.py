import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import foolbox as fb


attack_dict = {"FGSM_inf": fb.attacks.LinfFastGradientAttack,
               "FGSM_2": fb.attacks.L2FastGradientAttack,
               "FGSM_1": fb.attacks.L1FastGradientAttack,
               "PGD_2": fb.attacks.L2ProjectedGradientDescentAttack,
               "PGD_inf": fb.attacks.LinfProjectedGradientDescentAttack,
               "MIM_inf": fb.attacks.LinfBasicIterativeAttack}

__valid_tasks__ = ["imagenet", "cifar10", "mnist", "cifar100"]


class BetterDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class AdversarialAttack(nn.Module):
    def __init__(self, args, model, loader):
        super(AdversarialAttack, self).__init__()
        if args.dataset not in __valid_tasks__:
            raise NotImplementedError(f"Adversarial attacks are not implemented for {args.dataset}")

        self.attack_config = self._get_kwargs(args.loss_config_file)
        self.attack_fn = self._get_attack_fn()
        self.epsilon = [self.attack_config.epsilon]
        preprocessing = dict(mean=loader.dataset.means, std=loader.dataset.stds, axis=-3)
        self.fmodel = fb.PyTorchModel(model, bounds=loader.dataset.bounds, preprocessing=preprocessing)

    def _get_kwargs(self, config_file):
        with open(config_file, "r") as fp:
            attack_config = json.load(fp)
        attack_config = BetterDict(attack_config)
        return attack_config

    def _get_attack_fn(self):
        config = self.attack_config
        attack_cls = attack_dict["_".join([config.attack_name, config.constraint])]
        return attack_cls(**self.attack_config.attack_args)

    def forward(self, input, targets, logits, model):
        """
        Perturbs the input with gaussian noise, then evaluates kl-divergence between predictive distributions.
        """
        _, adv, _ = self.attack_fn(self.fmodel, input, targets, epsilons=self.epsilon)
        adv_logits = model(adv[0])
        adv_loss = F.nll_loss(F.log_softmax(adv_logits, dim=-1), targets)
        return adv_loss
