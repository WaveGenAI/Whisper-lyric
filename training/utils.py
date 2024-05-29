"""
This module contains the utilities for the training module.
"""
import glob
import warnings
from typing import Tuple

import librosa
import soundfile as sf
import torchaudio
from datasets import Dataset
from numpy import ndarray


def reformat_audio(path: str) -> Tuple[ndarray, int]:
    """
    Function that reformats the audio files.
    :param path: The path to the audio file.
    :return: The audio array and the sampling rate.
    """

    return


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
