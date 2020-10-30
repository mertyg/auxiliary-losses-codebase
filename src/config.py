import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description="Running loss term experiments.")
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--test-batch-size", default=None, type=int)
    parser.add_argument("--num-workers", default=None, type=int)
    parser.add_argument("--data-dir", default=None, type=str)
    parser.add_argument("--model", default="convnet")
    parser.add_argument(
        "--optimizer",
        default="Adam_0.001",
        help="Please pass this as OptimizerName_LR_Param1_Param2..",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-epochs", default=10)
    parser.add_argument("--eval-freq", default=1)
    parser.add_argument("--results-folder", default="./results")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument("--resume", default="", help="Location of the checkpoint file to resume training.")

    # Custom Loss arguments
    parser.add_argument("--custom-loss", default=None)
    parser.add_argument("--custom-loss-weight", default=1.)
    parser.add_argument("--loss-config-file")

    return parser
