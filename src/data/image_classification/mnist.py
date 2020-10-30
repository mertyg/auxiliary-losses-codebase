from torchvision import transforms
from torchvision.datasets import MNIST
import torch


def mnist(args):
    base_path = args.data_dir if args.data_dir else "./data-dir"
    batch_size = args.batch_size if args.batch_size else 256
    test_batch_size = args.test_batch_size if args.batch_size else 512
    num_workers = args.num_workers if args.num_workers else 4

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_data = MNIST(base_path, train=True, download=True,
                       transform=transform)
    test_data = MNIST(base_path, train=False,
                      transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size,
                                              shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
