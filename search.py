import math
import logging
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter, run_experiments
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from hyperpyyaml import load_hyperpyyaml
from utils import parse_arguments

logger = logging.getLogger(__name__)


def train_tune_checkpoint():
    pass

def main_tune(hparams):
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="val_acc",
        perturbation_interval=2,
        custom_explore_fn=None,  # this is the thing
        log_config = True
    )

    reporter = CLIReporter(
        parameter_columns=["lr"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])


    analysis = tune.run(
        tune.with_parameters(
            train_tune_checkpoint,
            num_epochs=20,
            num_gpus=1),

        resources_per_trial={
            "cpu": 2,
            "gpu": 1
        },
        metric="loss",
        mode="min",
        scheduler=pbt,
        rogress_reporter=reporter,
        name="tune_ecapa_tdnn"
    )

    print("Best augmentation schedules found were: ", analysis.best_config)


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logger.info("Starting...")

    # CLI:
    hparams_file, run_opts, overrides = parse_arguments(sys.argv[1:])

    with open(hparams_file) as fin:
        hparam_str = fin.read()

    if 'yaml' in run_opts:
        for yaml_file in run_opts['yaml']:
            logging.info(f"Loading additional yaml file: {yaml_file[0]}")
            with open(yaml_file[0]) as fin:
                hparam_str = hparam_str + "\n" + fin.read();

    hparams = load_hyperpyyaml(hparam_str, overrides)

    logging.info(f"Params: {hparams}")

    main_tune(hparams)


if __name__ == '__main__':
    main()
