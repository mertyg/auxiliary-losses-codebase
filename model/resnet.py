from torchvision.models.resnet import _resnet
from torchvision.models.resnet import Bottleneck, BasicBlock

__tasks__ = ["cifar10", "cifar100", "mnist", "imagenet"]
__resnet_dict__ = {"resnet50": ["resnet50", Bottleneck, [3, 4, 6, 3]],
                   "resnet18": ['resnet18', BasicBlock, [2, 2, 2, 2]]}


def _get_resnet(resnet_type, args, loader, pretrained=False, progress=True, **kwargs):
    dataset_type = loader.dataset.split("_")[0]
    if dataset_type not in __tasks__:
        raise NotImplementedError(f"ResNet is not implemented for {args.dataset}")

    return _resnet(*__resnet_dict__[resnet_type], pretrained, progress,
                   **kwargs)


def resnet50(args, loader, pretrained=False, progress=True, **kwargs):
    return _get_resnet("resnet50", args, loader, pretrained, progress, **kwargs)


def resnet18(args, loader, pretrained=False, progress=True, **kwargs):
    return _get_resnet("resnet18", args, loader, pretrained, progress, **kwargs)
