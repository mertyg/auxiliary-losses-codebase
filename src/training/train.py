import torch.nn.functional as F
from tqdm import tqdm


def train_step(experiment, args, model, loader, optimizer, custom_loss_fn, batch_wrap):
    # Loop adapted from https://github.com/pytorch/examples/blob/master/mnist/main.py
    model.train()
    b_loader = batch_wrap(loader)
    for (inputs, target) in b_loader:
        desc = []
        inputs, target = inputs.to(args.device), target.to(args.device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.nll_loss(F.log_softmax(logits, dim=-1), target)

        desc.append(f"Batch Loss: {loss.item():.2f}")
        if args.add_custom_loss:
            custom_loss = custom_loss_fn(inputs, logits, model)
            desc.append(f"Custom Loss: {custom_loss.item():.2f}")
            loss = loss + custom_loss

        if experiment.tqdm:
            b_loader.set_description(" ".join(desc))

        loss.backward()
        optimizer.step()
