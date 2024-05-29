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
        i = 0   # use to regenerate the dataset
        audios = glob.glob(path + "/audio/*")
        lyrics = glob.glob(path + "/lyrics/*.txt")
        for audio, lyric in zip(audios, lyrics):
            with open(lyrics[i], "r") as f:
                lyrics = f.read()
            yield {
                "audio": audio[i],
                "lyrics": lyrics,
            }
    return Dataset.from_generator(gen)
