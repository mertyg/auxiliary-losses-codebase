from torchvision import transforms
from torchvision import datasets
import torch
import os


def imagenet(args):
    base_path = args.data_dir if args.data_dir else "./data-dir"
    batch_size = args.batch_size if args.batch_size else 256
    test_batch_size = args.test_batch_size if args.batch_size else 1024
    num_workers = args.num_workers if args.num_workers else 4
    traindir = os.path.join(base_path, "imagenet", "train")
    valdir = os.path.join(base_path, "imagenet", "val")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_dataset.means = [0.485, 0.456, 0.406]
    train_dataset.stds = [0.229, 0.224, 0.225]
    train_dataset.bounds = [0, 1]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=test_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
