from data import get_dataset
from model import get_optimizer, get_model
from training import train_step, save_checkpoint, eval, plot_hist
from losses import get_loss
from config import arg_parser
from datetime import datetime
import os
import numpy as np
import json
import torch
from argparse import Namespace
from tqdm import tqdm


class Experiment:
    def __init__(self, args):
        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.optimizer = None
        self.custom_loss_fn = None
        self.init_experiment(args)
        self.args = args
        self.best_loss = np.inf
        self.best_epoch = -1
        self.epoch = 0
        self.log_file = os.path.join(args.logdir, "logs")
        self.tqdm = self.args.tqdm

        if self.args.tqdm:
            self.batch_wrap = tqdm
        else:
            self.batch_wrap = lambda x: x

        if self.args.resume:
            self.resume_training(self.args.resume)

        self.train_history = {}
        self.test_history = {}

    def init_experiment(self, args):
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%N")
        run_name = f"{args.dataset}_{args.model}_{args.optimizer}_{time}"
        folder = os.path.join(args.results_folder, run_name)
        os.makedirs(folder)  # Create the folders if it does not exist
        setattr(args, "logdir", folder)
        setattr(args, "checkpoint_path", os.path.join(folder, "ckpt"))
        setattr(args, "best_checkpoint", os.path.join(folder, "best_ckpt"))
        setattr(args, "plot_path", os.path.join(folder, "hist_{}.png"))
        train_loader, test_loader = get_dataset(args)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = get_model(args, train_loader).to(args.device)
        self.optimizer = get_optimizer(args, self.model)
        self.custom_loss_fn = get_loss(args, self.model, self.train_loader)

    def resume_training(self, resume_ckpt):
        checkpoint = torch.load(resume_ckpt)
        args = checkpoint["args"]
        args = Namespace(**args)
        self.args = args
        self.model = get_model(args).to(args.device)
        self.optimizer = get_optimizer(args, self.model)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.best_epoch = checkpoint["best_epoch"]
        print(f"Resuming from Epoch {self.epoch}")

    def report_logs(self, loss_dict):
        print(json.dumps(loss_dict))
        with open(self.log_file, "a") as fp:
            fp.write(json.dumps(loss_dict) + "\n")

    def update_history(self, loss_dict):
        if "t" not in self.train_history:
            self.train_history["t"] = []
            self.test_history["t"] = []
            for key in loss_dict["Test"].keys():
                self.train_history[key] = []
                self.test_history[key] = []

        self.train_history["t"].append(loss_dict["Epoch"])
        self.test_history["t"].append(loss_dict["Epoch"])

        for key in loss_dict["Test"].keys():
            self.train_history[key].append(loss_dict["Training"][key])
            self.test_history[key].append(loss_dict["Test"][key])

        plot_hist(self.args, self.train_history, self.test_history)

    def run(self):
        while self.epoch < self.args.max_epochs:
            train_step(self, self.args, self.model, self.train_loader, self.optimizer, self.custom_loss_fn, self.batch_wrap)
            self.epoch += 1

            if self.epoch % self.args.eval_freq == 0:
                save_checkpoint(self.model, self.epoch, self.optimizer, self.args, self.args.checkpoint_path)
                train_report = eval(self, self.model, self.train_loader, self.custom_loss_fn, self.args, self.batch_wrap)
                test_report = eval(self, self.model, self.test_loader, self.custom_loss_fn, self.args, self.batch_wrap)
                final_dict = {"Training": train_report, "Test": test_report}

                # Save as the best checkpoint
                if train_report["Loss"] < self.best_loss:
                    save_checkpoint(self.model, self.epoch, self.optimizer, self.args, self.args.best_checkpoint)
                    self.best_loss = train_report["Loss"]
                    self.best_epoch = self.epoch
                final_dict["Training"]["Best_Loss"] = self.best_loss
                final_dict["Training"]["Best_Epoch"] = self.best_epoch
                final_dict["Epoch"] = self.epoch
                self.report_logs(final_dict)
                self.update_history(final_dict)


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    experiment = Experiment(args)
    experiment.run()
