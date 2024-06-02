import argparse

from dataset.process import DatasetProcess


parser = argparse.ArgumentParser(
    description="Process the dataset",
)
parser.add_argument("--num_images", type=int, default=10000)
parser.add_argument("--clean", type=bool, default=True)

args = parser.parse_args()
api = 
