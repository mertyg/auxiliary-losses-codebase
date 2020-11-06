from .image_classification import cifar10, cifar100
from .image_classification import imagenet
from .image_classification import mnist
from .time_series import UCR
from torch.utils.data import DataLoader

__dataset_getters__ = {"cifar10": cifar10,
                       "cifar100": cifar100,
                       "mnist": mnist,
                       "imagenet": imagenet,
                       "UCR": UCR}


def get_dataset(args) -> (DataLoader, DataLoader):
    family = args.dataset.split("_")[0]
    if family not in __dataset_getters__:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")

    method = __dataset_getters__[family]
    return method(args)

