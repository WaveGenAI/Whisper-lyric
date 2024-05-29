from datasets import DatasetDict

from training.train import Trainer

import training.utils as utils

LOAD_DATASET = True

if LOAD_DATASET:
    dataset = utils.gather_dataset("./dataset")
    dataset = dataset.train_test_split(test_size=0.1)
else:
    dataset = DatasetDict.load_from_disk("./formated_dataset")
trainer = Trainer(dataset)
if LOAD_DATASET:
    dataset = trainer.process_dataset(dataset)
    dataset.save_to_disk("./formated_dataset")

trainer.train()
