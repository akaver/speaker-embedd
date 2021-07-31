import math
import logging
import random
import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter, run_experiments, Callback
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from hyperpyyaml import load_hyperpyyaml
from utils import parse_arguments
import ECAPA_TDNN
from data import VoxCeleb2Dataset
from main import SpeakerDataModule, EcapaTdnnModule
from pytorch_lightning.plugins import DDPPlugin
import augment

logger = logging.getLogger(__name__)

os.environ["SLURM_JOB_NAME"] = "bash"

def train_tune_checkpoint(config, hparams, checkpoint_dir=None):
    # send the augmentations to model pipeline
    hparams['augmentations'] = config['augmentations']

    print("train_tune_checkpoint", config)

    data = SpeakerDataModule(hparams)

    trainer = pl.Trainer(
        default_root_dir=hparams["data_folder"],
        gpus=-1,  # use all
        max_epochs=hparams["number_of_epochs"],
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=20,
        num_sanity_val_steps=0,
        precision=16,
        # accelerator='ddp',
        # plugins=DDPPlugin(find_unused_parameters=False),
        callbacks=[
            TuneReportCheckpointCallback(
                metrics={
                    "loss": "ptl/val_loss",
                    "mean_accuracy": "ptl/val_accuracy"
                },
                filename="checkpoint",
                on="validation_end")  # corresponds to the validation_epoch_end in lightning?
        ]
    )

    if checkpoint_dir:
        model = EcapaTdnnModule.load_from_checkpoint(
            os.path.join(checkpoint_dir, "checkpoint"),
            hparams=hparams, out_neurons=data.get_label_count())
    else:
        model = EcapaTdnnModule(hparams=hparams, out_neurons=data.get_label_count())

    trainer.fit(model, data)


class MyCallback(Callback):
    def on_trial_result(self, iteration, trials, trial, result,
                        **info):
        trial.config.update(result['config'])
        print(f"Got result: {result}")


def main_tune(hparams):
    def sample_augmentations():
        # how many augmentations to choose
        aug_count = random.randint(hparams["augmentations_min"], hparams["augmentations_max"])
        # random sampling without replacement
        sample = random.sample(hparams["augmentation_functions"], k=aug_count)
        return sample

    def explore(config):
        config["augmentations"] = sample_augmentations()
        return config

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",  # epoch count
        perturbation_interval=1,  # after what every time unit of time_attr to perturb
        # metric="loss", mode="min",
        metric="mean_accuracy", mode="max",
        custom_explore_fn=explore,  # called to produce the new augmentations perturbation
        log_config=True,
        # let the population to finish on same epoch, when next iteration start we have bigger space to choose from
        # since we use different number of augmentations, models will complete epoch on different duration
        synch=True,
        # copy top half to bottom half
        quantile_fraction=0.5
    )

    reporter = CLIReporter(
        parameter_columns=[],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            train_tune_checkpoint,
            hparams=hparams
        ),

        resources_per_trial={
            "cpu": 2,
            "gpu": 1.0 / 3.0
        },

        # callbacks=[MyCallback()],

        scheduler=pbt,
        num_samples=3,  # population size
        # how many times to train the full model, we perturb the augmentations on every model.epoch
        progress_reporter=reporter,
        name="tune_ecapa_tdnn",
        config={
            "augmentations": tune.sample_from(sample_augmentations)  # generate initial schedule step (dynamic)
        }

    )

    print("Best augmentation schedules found were: ", analysis)


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
