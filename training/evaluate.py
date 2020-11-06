import torch
import torch.nn.functional as F
from .helpers import AverageMeter, accuracy


def eval(experiment, model, loader, custom_loss_fn, args, batch_wrap):
    model.eval()
    custom_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        b_loader = batch_wrap(loader)
        for (inputs, targets) in b_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            N = targets.size(0)
            logits = model(inputs)
            loss = F.nll_loss(logits, targets)
            if logits.shape[1] <= 5:
                prec1 = accuracy(logits, targets, topk=(1,))
            else:
                prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
                top5.update(prec5[0], N)
            top1.update(prec1[0], N)
            losses.update(loss, N)

            if args.custom_loss:
                custom_loss = custom_loss_fn(inputs, targets, logits, model)
                custom_losses.update(custom_loss, N)

            if experiment.tqdm:
                if logits.shape[1] > 5:
                    b_loader.set_description(f"Top1: {top1.avg.item()}, Top5: {top5.avg}, Loss: {losses.avg.item()}, CustomLoss: {losses.avg.item()}")
                else:
                    b_loader.set_description(f"Top1: {top1.avg.item()}, Loss: {losses.avg.item()}, CustomLoss: {losses.avg.item()}")

    top1_acc = top1.avg
    loss = losses.avg
    try:
        custom_loss = custom_losses.avg.item()
    except:
        custom_loss = custom_losses.avg

    if logits.shape[1] > 5:
        top5_acc = top5.avg
        report = {"Top1": top1_acc.item(), "Top5": top5_acc.item(), "Loss": loss.item(), "CustomLoss": custom_loss}
    else:
        report = {"Top1": top1_acc.item(), "Loss": loss.item(), "CustomLoss": custom_loss}

    return report

