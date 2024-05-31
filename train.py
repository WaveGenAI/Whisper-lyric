from datasets import DatasetDict

from training.train import Trainer

from training import utils

LOAD_DATASET = True

if LOAD_DATASET:
    dataset = utils.gather_dataset("./dataset")
else:
    dataset = DatasetDict.load_from_disk("./formated_dataset")
trainer = Trainer(dataset)
if LOAD_DATASET:
    for i in range(dataset.num_rows//1000):
        dataset = trainer.process_dataset(dataset, i)
        dataset.save_to_disk(f"./formated_dataset_{i}")

trainer.train()
trainer.model.save_pretrained("./model")
trainer.processor.save_pretrained("./model")
