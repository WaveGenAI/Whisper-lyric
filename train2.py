from datasets import load_dataset, DatasetDict
from datasets import Audio

from training.train import Trainer

common_voice = DatasetDict()

common_voice["train"] = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "hi",
    split="train+validation",
    use_auth_token=True,
)
common_voice["test"] = load_dataset(
    "mozilla-foundation/common_voice_11_0", "hi", split="test", use_auth_token=True
)


common_voice = common_voice.remove_columns(
    [
        "accent",
        "age",
        "client_id",
        "down_votes",
        "gender",
        "locale",
        "path",
        "segment",
        "up_votes",
    ]
)

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

trainer = Trainer()


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = trainer.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = trainer.tokenizer(batch["sentence"]).input_ids
    return batch


common_voice = common_voice.map(
    prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1
)

trainer.train(common_voice)
