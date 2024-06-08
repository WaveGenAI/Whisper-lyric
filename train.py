from datasets import concatenate_datasets, Dataset

from training import utils
from training.train import Trainer
import argparse
import glob

parser = argparse.ArgumentParser(
    description="Process the dataset and train the model",
)

parser.add_argument(
    "--process_ds_path",
    type=str,
    default=None,
    help="Path to the dataset preprocessed",
)

parser.add_argument(
    "--chunked_ds_path",
    type=str,
    default=None,
    help="Path to the dataset chunked processed by this script",
)

parser.add_argument(
    "--model_path",
    type=str,
    default="./model",
    help="Path to save the model",
)

args = parser.parse_args()

if args.process_ds_path:
    dataset = utils.gather_dataset(args.process_ds_path)
    chuck_ds = []
    trainer = Trainer(dataset)

    ds = trainer.process_dataset(dataset)
    ds.save_to_disk(f"./dataset/process")
    dataset = ds
    trainer.dataset = dataset.train_test_split(test_size=0.3)
elif args.chunked_ds_path:
    dataset = Dataset.load_from_disk(f"{args.chunked_ds_path}")
    dataset = dataset.train_test_split(test_size=0.3)
    trainer = Trainer(dataset)
else:
    raise ValueError("You must provide either --process_ds_path or --chunked_ds_path")

print(dataset)

trainer.train()
trainer.save_model(args.model_path)
