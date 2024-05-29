from training.train import Trainer

import training.utils as utils


dataset = utils.gather_dataset("./dataset")
dataset = dataset.train_test_split(test_size=0.1)

trainer = Trainer(dataset)
trainer.process_dataset()
trainer.train()
