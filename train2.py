import librosa
from datasets import load_from_disk

from training import utils
from training.train import Trainer

DS_PATH = "dataset/export"

trainer = Trainer()

is_prepared = False


if not is_prepared:
    dataset = utils.gather_dataset(DS_PATH)
    target_sr = trainer.processor.feature_extractor.sampling_rate

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio, _ = librosa.load(batch["audio"], sr=target_sr)

        # compute log-Mel input features from input audio array
        batch["input_features"] = trainer.feature_extractor(
            audio, sampling_rate=target_sr
        ).input_features[0]

        # encode target text to label ids
        batch["labels"] = trainer.tokenizer(batch["lyrics"]).input_ids
        return batch

    dataset = dataset.map(
        prepare_dataset, remove_columns=dataset.column_names, num_proc=1
    )

    # filter out samples with empty labels
    dataset = dataset.filter(lambda x: len(x["labels"]) > 5)

    # save the processed dataset
    dataset.save_to_disk("dataset/test/")

else:
    # load the processed dataset
    dataset = load_from_disk("dataset/test/")

print(dataset)

dataset = dataset.train_test_split(test_size=0.05)
trainer.train(dataset)
