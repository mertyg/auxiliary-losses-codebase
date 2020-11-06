import torch.nn as nn
import torch.nn.functional as F
import torch
__valid_tasks_mlp__ = ["UCR"]
__valid_tasks_conv__ = ["mnist"]


class MLP(nn.Module):
    def __init__(self, args, loader, dropout_p=0.3):
        dataset_type = args.dataset.split("_")[0]
        if dataset_type not in __valid_tasks_mlp__:
            raise NotImplementedError(f"MLP is not implemented for {args.dataset}")
        dataset = loader.dataset
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dataset.input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(dropout_p)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, dataset.output_size)

    def forward(self, x, with_intermediate=False):
        out = x
        out = out.view(out.shape[0], -1)
        activations = []
        out = self.relu(self.fc1(out))
        activations.append(out)
        out = self.relu(self.fc2(self.dropout(out)))
        activations.append(out)
        out = self.fc3(out)
        if with_intermediate:
            return out, activations
        return out


class ConvNet(nn.Module):
    def __init__(self, args, loader):
        dataset_type = args.dataset.split("_")[0]
        if dataset_type not in __valid_tasks_conv__:
            raise NotImplementedError(f"ConvNet is not implemented for {args.dataset}")
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
