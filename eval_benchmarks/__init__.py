import torch.nn as nn
from .gaussian import Gaussian
from .eval_fn import eval_fn


__benchmarks__ = {"gaussian": Gaussian}


def get_benchmark(args, model, loader) -> nn.Module:
    if not args.eval_benchmark:
        return None
    benchmark_args = args.eval_benchmark.split("_")
    benchmark_type = benchmark_args[0]
    if benchmark_type not in __benchmarks__:
        raise NotImplementedError(f"{benchmark_type} benchmark is not implemented yet.")
    method = __benchmarks__[benchmark_type]
    return method(args, model, loader)
