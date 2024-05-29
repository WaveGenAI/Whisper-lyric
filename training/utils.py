"""

"""
import glob
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Dict, Union

import librosa
import soundfile as sf
from datasets import Dataset, Audio
from tqdm import tqdm


def reformat_audio():
    """Function that reformats the audio files."""
    print("Reformatting audio files...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        bar = tqdm(total=len(glob.glob("./dataset/audio/*")))
        for audio in glob.glob("./dataset/audio/*"):
            try:
                y, sr = sf.read(audio)
                if audio.split(".")[-1] != "ogg" and sr != 16000:
                    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                    audio = audio.replace(extension, "ogg")
                    sf.write(audio, y, sr, format='ogg', subtype='vorbis')
            except Exception:
                extension = audio.split(".")[-1]
                y, sr = librosa.load(audio, sr=None)
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                audio = audio.replace(extension, "ogg")
                sf.write(audio, y, sr, format='ogg', subtype='vorbis')
            bar.update(1)


def gather_dataset(path: str) -> Dataset:
    """Function that gathers the dataset.

    Args:
    path (str): The path to the dataset.

    Returns:
    Dataset: The dataset.
    """
    def gen():
        i = 0
        audio = glob.glob(path + "/audio/*.ogg")
        lyrics = glob.glob(path + "/lyrics/*.txt")
        for i in range(len(lyrics)):
            yield {
                "audio": audio[i],
                "lyrics": open(lyrics[i], "r").read(),
            }
    # reformat_audio()
    return Dataset.from_generator(gen).cast_column("audio", Audio())