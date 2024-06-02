import argparse

from dataset.process import DatasetProcess


parser = argparse.ArgumentParser(
    description="Process the dataset",
)
parser.add_argument("--audio_path", type=str, default="dataset/audio")
parser.add_argument("--lyric_path", type=str, default="dataset/lyrics")
parser.add_argument("--export_path", type=str, default="dataset/export")
parser.add_argument("--clean", type=bool, default=False)

args = parser.parse_args()
process = DatasetProcess(
    lyric_path=args.lyric_path,
    audio_path=args.audio_path,
    export_path=args.export_path,
    clean=args.clean,
)

process.process()
