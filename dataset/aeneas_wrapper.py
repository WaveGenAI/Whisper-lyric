import json
import re
import tempfile

from aeneas.tools.execute_task import ExecuteTaskCLI, RuntimeConfiguration

from dataset.exceptions import AeneasAlignError


class AeneasWrapper:
    """Wrapper class for Aeneas CLI"""

    def __init__(self) -> None:
        self._rconf = RuntimeConfiguration()
        self._rconf[RuntimeConfiguration.MFCC_MASK_NONSPEECH] = True
        self._rconf[RuntimeConfiguration.MFCC_MASK_NONSPEECH_L3] = True
        self._rconf[RuntimeConfiguration.TTS_CACHE] = True
        self._rconf.set_granularity(3)

    def aeneas_cli_exec(self, audio_path: str, lyric_path: str) -> dict:
        """Align lyrics with audio

        Args:
            audio_path (str): the path to the audio file
            lyric_path (str): the path to the lyric file

        Raises:
            AeneasAlignError: if Aeneas fails to align lyrics

        Returns:
            dict: a dictionary containing the alignment data
        """

        tmp_dir = tempfile.mkdtemp()

        with open(lyric_path, "r", encoding="utf-8") as f:
            lyric = f.read()

        # remove all text between []
        lyric = re.sub(r"\[.*?\]", "\n", lyric)

        # remove when more than 2 new lines
        lyric = re.sub(r"\n{1,}", "\n", lyric).strip()

        lyric = lyric.replace(" ", "\n")

        with open(f"{tmp_dir}/lyric.txt", "w", encoding="utf-8") as f:
            f.write(lyric)

        args = [
            "dummy",
            audio_path,
            f"{tmp_dir}/lyric.txt",
            "task_language=en|is_text_type=plain|os_task_file_format=json",
            f"{tmp_dir}/lyric.json",
        ]

        exit_code = ExecuteTaskCLI(use_sys=False, rconf=self._rconf).run(arguments=args)

        if exit_code != 0:
            raise AeneasAlignError("Aeneas failed to align lyrics")

        with open(f"{tmp_dir}/lyric.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        return data
