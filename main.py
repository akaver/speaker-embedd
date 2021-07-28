import logging
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from utils import parse_arguments
from hyperpyyaml import load_hyperpyyaml
from pytorch_lightning.plugins import DDPPlugin
from torchmetrics.functional import accuracy
import ECAPA_TDNN
from data import VoxCeleb2Dataset

logger = logging.getLogger(__name__)


class SpeakerDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()

        self.batch_size = hparams["batch_size"]
        self.num_workers = hparams["num_workers"]

        # assign to use in dataloaders

        self.train_dataset = VoxCeleb2Dataset(hparams, hparams["train_data"])
        self.val_dataset = VoxCeleb2Dataset(hparams, hparams["valid_data"])
        self.test_dataset = VoxCeleb2Dataset(hparams, hparams["test_data"])
        self.enrol_dataset = VoxCeleb2Dataset(hparams, hparams["enrol_data"])

    """
    # Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.
    def prepare_data(self):
        pass

    # There are also data operations you might want to perform on every GPU
    def setup(self, stage = None):
        pass
    """

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True)

    def enrol_dataloader(self):
        return DataLoader(self.enrol_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True,
                          persistent_workers=True)

    """
    def test_dataloader(self):
        pass

    # ow you want to move an arbitrary batch to a device. on single gpu only
    def transfer_batch_to_device(self, batch, device):
        pass

    # alter or apply augmentations to your batch before it is transferred to the device.
    def on_before_batch_transfer(self, batch, dataloader_idx):
        pass

    #  alter or apply augmentations to your batch after it is transferred to the device.
    def on_after_batch_transfer(self, batch, dataloader_idx):
        pass

    #  can be used to clean up the state. It is also called from every process
    def teardown(self, stage = None):
        pass
    """

    def get_label_count(self):
        return self.train_dataset.get_label_count()


class EcapaTdnnModule(pl.LightningModule):
    def __init__(self, hparams, out_neurons):
        super(EcapaTdnnModule, self).__init__()

        # naming conflict?
        self._hparams = hparams

        self.out_neurons = out_neurons

        # embedding size
        lin_neurons = 192

        self.compute_features = ECAPA_TDNN.Fbank(
            n_mels=hparams["n_mels"],
            left_frames=hparams["left_frames"], right_frames=hparams["right_frames"],
            deltas=hparams["deltas"])

        self.mean_var_norm = ECAPA_TDNN.InputNormalization(norm_type="sentence", std_norm=False)

        self.net = ECAPA_TDNN.ECAPA_TDNN(input_size=int(hparams["n_mels"]), lin_neurons=lin_neurons)

        # embedding in, speaker count out
        self.classifier = ECAPA_TDNN.Classifier(lin_neurons, out_neurons=out_neurons)

        self.compute_cost = ECAPA_TDNN.LogSoftmaxWrapper(loss_fn=ECAPA_TDNN.AdditiveAngularMargin(margin=0.2, scale=30))
        # not used
        self.compute_error = ECAPA_TDNN.classification_error

    def prepare_data(self):
        pass

    def setup(self, stage):
        pass

    # Use for inference only (separate from training_step)
    def forward(self, wavs):
        # wavs is batch of tensors with audio

        lens = torch.ones(wavs.shape[0])

        wavs_aug_tot = []
        wavs_aug_tot.append(wavs)


        # TODO - apply augmentations here!
        self.n_augment = 1


        wavs = torch.cat(wavs_aug_tot, dim=0)
        lens = torch.cat([lens] * self.n_augment)


        # extract features - fliterbanks from mfcc
        features = self.compute_features(wavs)

        # normalize
        features_normalized = self.mean_var_norm(features, lens)

        embedding = self.net(features_normalized)

        prediction = self.classifier(embedding)
        return prediction, embedding

    def get_embeddings(self, wavs):
        _, embeddings = self.forward(wavs)
        return embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.000002)
        return optimizer
        pass

    def model_step(self, batch, batch_idx):
        inputs, labels, ids = batch
        labels_predicted = self(inputs)  # calls forward
        loss = self.compute_cost(labels_predicted, labels)

        labels_predicted_squeezed = labels_predicted.squeeze()
        labels_squeezed = labels.squeeze()

        labels_hot_encoded = F.one_hot(labels_squeezed.long(), labels_predicted_squeezed.shape[1])
        acc = accuracy(labels_predicted_squeezed, labels_hot_encoded)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.model_step(batch, batch_idx)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self.model_step(batch, batch_idx)
        return {'loss': loss, 'acc': acc}

    def test_step(self, batch, batch_idx):
        # return Union[Tensor, Dict[str, Any], None]
        pass

    # Called at the end of the validation epoch with the outputs of all validation steps.
    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['acc'] for x in val_step_outputs]).mean()

        self.log('avg_val_loss', avg_val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('avg_val_acc', avg_val_acc, on_epoch=True, prog_bar=True, logger=True)

        logger.info(f"avg_val_loss: {avg_val_loss} avg_val_acc: {avg_val_acc}")
        return {'val_loss': avg_val_loss}


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logger.info("Starting...")

    # enable the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

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

    # =============================== Pytorch Lightning ====================================

    data = SpeakerDataModule(hparams)

    logger.info(f"Speakers found {data.get_label_count()}")

    model = EcapaTdnnModule(hparams, out_neurons=data.get_label_count())

    trainer = pl.Trainer(
        default_root_dir=hparams["data_folder"],
            gpus=1,
        max_epochs=hparams["number_of_epochs"],
        # num_sanity_val_steps=0
        # accelerator='ddp',
        # plugins=DDPPlugin(find_unused_parameters=False)
    )

    trainer.fit(model, data)


if __name__ == '__main__':
    main()
