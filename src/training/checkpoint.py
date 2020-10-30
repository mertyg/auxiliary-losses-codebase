import torch


def save_checkpoint(model, epoch, optimizer, args, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint():
    pass
