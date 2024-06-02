from training import utils
from training.train import Trainer

dataset = utils.gather_dataset("./dataset/export")
dataset = dataset.train_test_split(test_size=0.1)

trainer = Trainer(dataset)
dataset = trainer.process_dataset(dataset)

trainer.train()
