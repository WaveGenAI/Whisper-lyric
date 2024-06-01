import os

import dataset.exceptions
from dataset.aeneas_wrapper import aeneas_cli_exec


class Process:
    """Class to process the dataset"""

    def __init__(self, lyric_path: str, audio_path: str):
        self.lyric_path = lyric_path
        self.audio_path = audio_path

    def _aenas_align(self, audio_path: str, lyric_path: str) -> dict:
        """Method to align lyrics with audio

        Args:
            audio_path (str): the path to the audio file
            lyric_path (str): the path to the lyric file

        Raises:
            AeneasAlignError: if Aeneas fails to align lyrics

        Returns:
            dict: a dictionary containing the alignment data
        """

        return aeneas_cli_exec(audio_path, lyric_path)

    def _split_audio(self, lyric_path: str, alignement: dict) -> list:
        """Method to split audio into 32 seconds segments with the corresponding lyrics

        Args:
            lyric_path (str): the path to the lyric file
            alignement (dict): the alignment data

        Returns:
            list: a list that contain lyrics split into 32 seconds segments
        """

        raise NotImplementedError

    def process(self) -> None:
        """Method to process the dataset :
        1. Align lyrics with audio
        2. Split audio into 32 seconds segments
        3. Save the segments to the dataset/audio/processed folder in .wav format
        """

        for audio_f in os.listdir(self.audio_path):
            audio_path = os.path.join(self.audio_path, audio_f)
            lyric_path = os.path.join(self.lyric_path, audio_f.split(".")[0] + ".txt")

            try:
                alignement = self._aenas_align(audio_path, lyric_path)
            except dataset.exceptions.AeneasAlignError as e:
                print(f"Failed to align {audio_f}: {e}")
                continue
