from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import torch


def cifar10(args):
    base_path = args.data_dir if args.data_dir else "./data-dir"
    batch_size = args.batch_size if args.batch_size else 64
    test_batch_size = args.test_batch_size if args.batch_size else 128
    num_workers = args.num_workers if args.num_workers else 2

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(root=base_path, train=True,
                       download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)

    testset = CIFAR10(root=base_path, train=False,
                      download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                              shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def cifar100(args):
    base_path = args.data_dir if args.data_dir else "./data-dir"
    batch_size = args.batch_size if args.batch_size else 64
    test_batch_size = args.test_batch_size if args.batch_size else 128
    num_workers = args.num_workers if args.num_workers else 2

    train_loader = torch.utils.data.DataLoader(
                            CIFAR100(base_path, train=True, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                            batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
                            CIFAR100(base_path, train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                            batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
