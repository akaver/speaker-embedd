import os
import argparse
import torch
import logging

logger = logging.getLogger(__name__)


def parse_arguments(arg_list):
    r"""Parse command-line arguments to the experiment.

    Arguments
    ---------
    arg_list : list
        A list of arguments to parse, most often from `sys.argv[1:]`.

    Returns
    -------
    param_file : str or [[str, str, ...]]
        The location of the parameters file.
    run_opts : dict
        Run options, such as distributed, device, etc.
    overrides : dict
        The overrides to pass to ``load_hyperpyyaml``.

    Example
    -------
    >>> argv = ['hyperparams.yaml', '--device', 'cuda:1', '--seed', '10']
    >>> filename, run_opts, overrides = parse_arguments(argv)
    >>> filename
    'hyperparams.yaml'
    >>> run_opts["device"]
    'cuda:1'
    >>> overrides
    'seed: 10'
    """
    parser = argparse.ArgumentParser(
        description="Speaker embedding",
    )

    parser.add_argument(
        "param_file",
        type=str,
        help="A yaml-formatted file using the extended YAML syntax. "
             "defined by SpeechBrain.",
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Run the experiment with only a few batches for all "
             "datasets, to ensure code runs without crashing.",
    )
    parser.add_argument(
        "--debug_batches",
        type=int,
        default=2,
        help="Number of batches to run in debug mode.",
    )
    parser.add_argument(
        "--debug_epochs",
        type=int,
        default=2,
        help="Number of epochs to run in debug mode. "
             "If a non-positive number is passed, all epochs are run.",
    )
    parser.add_argument(
        "--log_config",
        type=str,
        help="A file storing the configuration options for logging",
    )
    # if use_env = False in torch.distributed.lunch then local_rank arg is given
    parser.add_argument(
        "--local_rank", type=int, help="Rank on local machine",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device to run the experiment on (e.g. 'cuda:0')",
    )
    parser.add_argument(
        "--data_parallel_backend",
        default=False,
        action="store_true",
        help="This flag enables training with data_parallel.",
    )
    parser.add_argument(
        "--distributed_launch",
        default=False,
        action="store_true",
        help="This flag enables training with DDP. Assumes script run with "
             "`torch.distributed.launch`",
    )
    parser.add_argument(
        "--distributed_backend",
        type=str,
        default="nccl",
        help="One of {nccl, gloo, mpi}",
    )
    parser.add_argument(
        "--find_unused_parameters",
        default=False,
        action="store_true",
        help="This flag disable unused parameters detection",
    )
    parser.add_argument(
        "--jit_module_keys",
        type=str,
        nargs="*",
        help="A list of keys in the 'modules' dict to jitify",
    )
    parser.add_argument(
        "--auto_mix_prec",
        default=False,
        action="store_true",
        help="This flag enables training with automatic mixed-precision.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        help="Gradient norm will be clipped to this value, "
             "enter negative value to disable.",
    )
    parser.add_argument(
        "--nonfinite_patience",
        type=int,
        help="Max number of batches per epoch to skip if loss is nonfinite.",
    )
    parser.add_argument(
        "--noprogressbar",
        default=False,
        action="store_true",
        help="This flag disables the data loop progressbars.",
    )
    parser.add_argument(
        "--ckpt_interval_minutes",
        type=float,
        help="Amount of time between saving intra-epoch checkpoints "
             "in minutes. If non-positive, intra-epoch checkpoints are not saved.",
    )

    parser.add_argument(
        "--yaml",
        type=str,
        action='append',
        nargs='+',
        help="A yaml-formatted file using the extended YAML syntax. "
             "Multiple files loaded in order on top of required yaml",
    )

    # Accept extra args to override yaml
    run_opts, overrides = parser.parse_known_args(arg_list)

    # Ignore items that are "None", they were not passed
    run_opts = {k: v for k, v in vars(run_opts).items() if v is not None}

    param_file = run_opts["param_file"]
    del run_opts["param_file"]

    overrides = _convert_to_yaml(overrides)

    # Checking that DataParallel use the right number of GPU
    if run_opts["data_parallel_backend"]:
        if torch.cuda.device_count() == 0:
            raise ValueError("You must have at least 1 GPU.")

    # For DDP, the device args must equal to local_rank used by
    # torch.distributed.launch. If run_opts["local_rank"] exists,
    # use os.environ["LOCAL_RANK"]
    local_rank = None
    if "local_rank" in run_opts:
        local_rank = run_opts["local_rank"]
    else:
        if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] != "":
            local_rank = int(os.environ["LOCAL_RANK"])

    # force device arg to be the same as local_rank from torch.distributed.lunch
    if local_rank is not None and "cuda" in run_opts["device"]:
        run_opts["device"] = run_opts["device"][:-1] + str(local_rank)

    return param_file, run_opts, overrides


def _convert_to_yaml(overrides):
    """Convert args to yaml for overrides"""
    yaml_string = ""

    # Handle '--arg=val' type args
    joined_args = "=".join(overrides)
    split_args = joined_args.split("=")

    for arg in split_args:
        if arg.startswith("--"):
            yaml_string += "\n" + arg[len("--"):] + ":"
        else:
            yaml_string += " " + arg

    return yaml_string.strip()
