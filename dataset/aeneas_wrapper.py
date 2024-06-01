import json
import tempfile

from aeneas.tools.execute_task import ExecuteTaskCLI

from dataset.exceptions import AeneasAlignError


def aeneas_cli_exec(audio_path: str, lyric_path: str) -> dict:
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

    args = [
        "dummy",
        audio_path,
        lyric_path,
        "task_language=en|is_text_type=plain|os_task_file_format=json",
        f"{tmp_dir}/lyric.json",
    ]

    exit_code = ExecuteTaskCLI(use_sys=False).run(arguments=args)

    if exit_code != 0:
        raise AeneasAlignError("Aeneas failed to align lyrics")

    with open(f"{tmp_dir}/lyric.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    return data
