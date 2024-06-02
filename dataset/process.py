import os

from pydub import AudioSegment

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

    def _split_audio(
        self, lyric_path: str, alignement: dict, split_windows: int = 32
    ) -> list:
        """Method to split audio into 32 seconds segments with the corresponding lyrics

        Args:
            lyric_path (str): the path to the lyric file
            alignement (dict): the alignment data
            split_windows (int, optional): the size of the split window in seconds. Defaults to 32.

        Returns:
            list: a list of list that contain lyrics split into 32 seconds segments
        """

        lyric = open(lyric_path, "r", encoding="utf-8").read()

        segments = []
        start_idx = 0
        end_idx = 0

        for fragment in alignement["fragments"]:
            print(fragment)
            end_idx = lyric.find(fragment["lines"][0], start_idx)
            windows = (len(segments) + 1) * split_windows

            if float(fragment["begin"]) > windows:
                segments.append(lyric[start_idx:end_idx])
                start_idx = end_idx

        segments.append(lyric[start_idx:])

        print(segments, len(segments))

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

            self._split_audio(lyric_path, alignement)

            break
