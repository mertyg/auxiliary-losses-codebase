from data import get_dataset
from model import get_model
from config import eval_arg_parser
from eval_benchmarks import eval_fn, get_benchmark
import os
import torch
from argparse import Namespace
from tqdm import tqdm
import json


class EvalExperiment:
    def __init__(self, args):
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.benchmark_fn = None
        self.init_experiment(args)
        self.args = args
        self.epoch = None
        self.log_file = os.path.join(args.logdir, f"logs-{self.args.eval_benchmark}-{self.args.benchmark_config}")
        self.tqdm = self.args.tqdm

        if self.args.tqdm:
            self.batch_wrap = tqdm
        else:
            self.batch_wrap = lambda x: x

        self.load_ckpt(self.args.best_checkpoint)

    def init_experiment(self, args):
        folder = args.eval_run
        setattr(args, "logdir", folder)
        setattr(args, "checkpoint_path", os.path.join(folder, "ckpt"))
        setattr(args, "best_checkpoint", os.path.join(folder, "best_ckpt"))
        train_loader, test_loader = get_dataset(args)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = get_model(args, train_loader).to(args.device)
        self.benchmark_fn = get_benchmark(args, self.model, self.train_loader)

    def report_results(self, report):
        print(json.dumps(report))
        with open(self.log_file, "a") as fp:
            fp.write(json.dumps(report) + "\n")

    def load_ckpt(self, resume_ckpt):
        checkpoint = torch.load(resume_ckpt)
        args = checkpoint["args"]
        args = Namespace(**args)
        self.args = args
        self.model = get_model(args, self.train_loader).to(args.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.epoch = checkpoint["epoch"]
        print(f"Evaluation Epoch {self.epoch}")

    def run(self):
        train_report = eval_fn(self, self.model, self.train_loader, self.benchmark_fn, self.args, self.batch_wrap)
        test_report = eval_fn(self, self.model, self.test_loader, self.benchmark_fn, self.args, self.batch_wrap)
        report = {"Train": train_report, "Test": test_report}
        self.report_results(report)


if __name__ == "__main__":
    parser = eval_arg_parser()
    args = parser.parse_args()
    experiment = EvalExperiment(args)
    experiment.run()
