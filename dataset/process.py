import os
import shutil
from typing import List

from pydub import AudioSegment
from tqdm import tqdm

import dataset.exceptions
from dataset.aeneas_wrapper import AeneasWrapper


class DatasetProcess:
    """Class to process the dataset"""

    def __init__(
        self,
        lyric_path: str,
        audio_path: str,
        sample_rate: int = None,
        export_path: str = None,
        clean: bool = False,
    ):
        """Constructor to initialize the DatasetProcess class

        Args:
            lyric_path (str): the path to the lyrics folder
            audio_path (str): the path to the audio folder
            sample_rate (int, optional): the sample rate of the audio. Defaults to None.
            export_path (str, optional): the path to export data. Defaults to None.
            clean (bool, optional): remove all data in the export path. Defaults to False.
        """

        self.lyric_path = lyric_path
        self.audio_path = audio_path
        self.export_path = export_path
        self.sample_rate = sample_rate

        if clean and self.export_path:
            self.remove_export_folder()

        self.create_export_folder()

        self.aeneas = AeneasWrapper()

    def create_export_folder(self) -> None:
        """Method to create the export folder"""

        if not os.path.exists(self.export_path):
            os.makedirs(self.export_path)

        if not os.path.exists(f"{self.export_path}/audio"):
            os.makedirs(f"{self.export_path}/audio")

        if not os.path.exists(f"{self.export_path}/lyrics"):
            os.makedirs(f"{self.export_path}/lyrics")

    def remove_export_folder(self) -> None:
        """Method to remove the export folder"""

        if os.path.exists(self.export_path):
            shutil.rmtree(self.export_path)

    def _split_audio(
        self, audio_path: str, split_windows: int = 30
    ) -> List[AudioSegment]:
        """Method to split audio into 32 seconds segments

        Args:
            audio_path (str): the path to the audio file
            split_windows (int, optional): the size of the split window in seconds. Defaults to 32.

        Returns:
            list: a list of AudioSegment that contain audio split into 32 seconds segments
        """

        audio = AudioSegment.from_file(audio_path)
        segments = []

        for i in range(0, len(audio), split_windows * 1000):
            segments.append(audio[i : i + split_windows * 1000])

        return segments

    def _split_lyric(
        self, lyric_path: str, alignement: dict, split_windows: int = 30
    ) -> List[str]:
        """Method to split audio into 32 seconds segments with the corresponding lyrics

        Args:
            lyric_path (str): the path to the lyric file
            alignement (dict): the alignment data
            split_windows (int, optional): the size of the split window in seconds. Defaults to 32.

        Returns:
            list: a list of list that contain lyrics split into 32 seconds segments
        """

        with open(lyric_path, "r", encoding="utf-8") as f:
            lyric = f.read()

        segments = []
        start_idx = 0
        end_idx = 0

        for fragment in alignement["fragments"]:
            end_idx = lyric.find(fragment["lines"][0], end_idx)
            windows = (len(segments) + 1) * split_windows

            if float(fragment["begin"]) > windows:
                segments.append(lyric[start_idx:end_idx])
                start_idx = end_idx

        segments.append(lyric[start_idx:])

        return segments

    def _export_audio(self, audios: List[AudioSegment], file_name: str) -> None:
        """Method to export audio segments to .wav format

        Args:
            audios (List[AudioSegment]): a list of AudioSegment
            file_name (str): the name of the file
        """

        for i, audio in enumerate(audios):
            path = f"{self.audio_path}/{file_name}_{i}.wav"

            if self.export_path:
                path = f"{self.export_path}/audio/{file_name}_{i}.wav"

            if self.sample_rate:
                audio = audio.set_frame_rate(self.sample_rate)

            audio.export(path, format="wav")

    def _export_lyric(self, lyrics: List[str], file_name: str) -> None:
        """Method to export lyrics segments to .txt format

        Args:
            lyrics (List[str]): a list of lyrics
            file_name (str): the name of the file
        """

        for i, lyric in enumerate(lyrics):
            path = f"{self.lyric_path}/{file_name}_{i}.txt"

            if self.export_path:
                path = f"{self.export_path}/lyrics/{file_name}_{i}.txt"

            with open(path, "w", encoding="utf-8") as f:
                f.write(lyric)

    def process(self, remove: bool = False) -> None:
        """Method to process the dataset
            1. Align lyrics with audio
            2. Split audio into 32 seconds segments
            3. Save the segments to the dataset/audio/processed folder in .wav format

        Args:
            remove (bool, optional): remove the processed file. Defaults to False.
        """

        nbm_files = len(os.listdir(self.audio_path))
        progress_bar = tqdm(total=nbm_files)
        for i, audio_f in enumerate(os.listdir(self.audio_path)):
            if not audio_f.endswith(".ogg") and not audio_f.endswith(".mp4"):
                continue

            audio_path = os.path.join(self.audio_path, audio_f)
            lyric_path = os.path.join(self.lyric_path, audio_f.split(".")[0] + ".txt")

            try:
                alignement = self.aeneas.aeneas_cli_exec(audio_path, lyric_path)
            except dataset.exceptions.AeneasAlignError as e:
                print(f"Failed to align {audio_f}: {e}")
                continue

            lyric_segments = self._split_lyric(lyric_path, alignement)
            audio_segments = self._split_audio(audio_path)

            # save the audio segments and the lyrics
            self._export_audio(audio_segments, audio_f.split(".")[0])
            self._export_lyric(lyric_segments, audio_f.split(".")[0])

            # print(
            #     f"Processed {i}/ {nbm_files} - {round(i/nbm_files*100, 2)}%", end="\r"
            # )

            if remove:
                os.remove(lyric_path)
                os.remove(audio_path)

            progress_bar.update(1)
        progress_bar.close()
