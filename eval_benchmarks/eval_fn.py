import torch
from robustness.tools.helpers import AverageMeter


def init_report(batch_report):
    report = {}
    for metric, val in batch_report.items():
        report[metric] = AverageMeter()
    return report


def update_report(report, batch_report, N):
    for metric, val in batch_report.items():
        report[metric].update(val, N)


def eval_fn(experiment, model, loader, benchmark_fn, args, batch_wrap):
    model.eval()
    report = None
    with torch.no_grad():
        b_loader = batch_wrap(loader)
        for (inputs, targets) in b_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            logits = model(inputs)
            N = targets.size(0)
            batch_report = benchmark_fn(inputs, targets, logits, model)
            if report is None:
                report = init_report(batch_report)
            update_report(report, batch_report, N)
            if experiment.tqdm:
                desc = [f"{key}: {val.avg.item()}" for key, val in report.items()]
                b_loader.set_description(" ".join(desc))

    for key, val in report.items():
        report[key] = val.avg.item()

    return report

