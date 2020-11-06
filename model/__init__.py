from .basic import MLP, ConvNet
from .resnet import resnet50
from .resnet import resnet18

import torch.optim as optim
import torch.nn as nn

__models__ = {"mlp": MLP, "resnet50": resnet50, "resnet18": resnet18, "convnet": ConvNet}


def get_model(args, loader) -> nn.Module:
    if args.model.lower() not in __models__:
        raise NotImplementedError(f"{args.model} is not implemented yet.")
    model = __models__[args.model.lower()](args, loader)
    return model


def get_optimizer(args, model):
    optim_config = args.optimizer.split("_")
    optim_name = optim_config[0]
    optim_params = [float(p) for p in optim_config[1:]]

    if optim_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), *optim_params)

    elif optim_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), *optim_params)

    else:
        raise NotImplementedError(f"{optim_name} is not implemented.")

    return optimizer
