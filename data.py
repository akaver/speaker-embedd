import torch
import utils
import logging
import sys
import time
import random
import ECAPA_TDNN
from tqdm import tqdm
from utils import parse_arguments
from hyperpyyaml import load_hyperpyyaml
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)


class VoxCeleb2Dataset(torch.utils.data.Dataset):
    def __init__(self, hparams, csv_data_file):
        super(VoxCeleb2Dataset, self).__init__()
        self.hparams = hparams
        self.snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

        # dictionary of id:str - {duration:float, wav:str, start:int, stop:int, spk_id:str}
        self.csv_data = utils.load_data_csv(csv_data_file)

        self.data = []

        # get the labels
        self.labels = {}
        for key in self.csv_data:
            self.labels[self.csv_data[key]["spk_id"]] = None
            self.data.append(key)

        self.labels = list(self.labels.keys())

    def __getitem__(self, index):

        item = self.csv_data[self.data[index]]

        # read audio
        start = item["start"]
        stop = item["stop"]
        duration = item["duration"]

        # tensor with wav
        wav_tensor = self.load_audio(item["wav"], start, stop, duration)

        # return audio_sample, y
        # one hot encoding for label
        # y = self.to_categorical(self.labels.index(item["spk_id"]), len(self.labels))

        # use indexes instead - hot encode them in loss fn - label count from here needs to match final output neurons
        y = torch.tensor([self.labels.index(item["spk_id"])])

        res = wav_tensor,  y
        return res

    def __len__(self):
        return len(self.data)

    def get_label_count(self):
        return len(self.labels)

    def load_audio(self, wav, start, stop, duration):

        # instead of predefined chunks pick random chunk over the whole clip
        if self.hparams["random_chunk"]:
            duration_sample = int(duration * self.hparams["sample_rate"])
            start = random.randint(0, duration_sample - self.snt_len_sample - 1)
            stop = start + self.snt_len_sample

        num_frames = stop - start
        # Resulting Tensor and sample rate.
        # If the input file has integer wav format and normalization is off, then it has integer type, else float32 type.
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    @staticmethod
    def to_categorical(class_num, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[class_num]


def main():
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
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

    voxceleb_dataset_train = VoxCeleb2Dataset(hparams, hparams["train_data"])
    voxceleb_dataset_val = VoxCeleb2Dataset(hparams, hparams["valid_data"])

    data_loader_train = torch.utils.data.DataLoader(voxceleb_dataset_train, batch_size=12, shuffle=True,
                                                    num_workers=4, pin_memory=True,
                                                    )
    data_loader_val = torch.utils.data.DataLoader(voxceleb_dataset_val, batch_size=12, shuffle=False,
                                                  num_workers=4, pin_memory=True,
                                                  )

    for i_batch, sample_batched in enumerate(data_loader_train):
        print(i_batch, len(sample_batched), len(sample_batched[0]), len(sample_batched[1]))
        print(sample_batched)

    """
    iterator_train = iter(data_loader_train)
    iterator_val = iter(data_loader_val)
    e = time.time()
    for i in tqdm(range(20)):
        a, b = next(iterator_train)
        print(a,b)
        # print("\n")
        # print(i, a[0].shape, b[0].shape, time.time() - e)
        e = time.time()

    e = time.time()
    for i in tqdm(range(20)):
        a, b = next(iterator_val)
        print(a,b)
        # print("\n")
        # print(i, a[0].shape, b[0].shape, time.time() - e)
        e = time.time()
    """

if __name__ == "__main__":
    main()
