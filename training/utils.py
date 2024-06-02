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
        audios = glob.glob(path + "/audio/*")
        lyrics = glob.glob(path + "/lyrics/*.txt")
        for audio, lyric in zip(audios, lyrics):
            with open(lyric, "r", encoding="utf-8") as f:
                lyric = f.read()
            yield {
                "audio": audio,
                "lyrics": lyric,
            }

    return Dataset.from_generator(gen)
