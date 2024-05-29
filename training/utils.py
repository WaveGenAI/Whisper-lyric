"""
This module contains the utilities for the training module.
"""
import glob

from datasets import Dataset


def gather_dataset(path: str) -> Dataset:
    """Function that gathers the dataset.

    Args:
    path (str): The path to the dataset.

    Returns:
    Dataset: The dataset.
    """
    def gen():
        i = 1 # use to
        audio = glob.glob(path + "/audio/*")
        lyrics = glob.glob(path + "/lyrics/*.txt")
        for i in range(len(lyrics)):
            yield {
                "audio": audio[i],
                "lyrics": open(lyrics[i], "r").read(),
            }
    return Dataset.from_generator(gen)
