import argparse

from dataset.build import SonautoAPI


parser = argparse.ArgumentParser(
    description="Download images from Sonauto dataset",
)
parser.add_argument("--num_images", type=int, default=1000)
parser.add_argument("--clean", type=bool, default=True)

args = parser.parse_args()
api = SonautoAPI(clean=args.clean)
api.download(args.num_images)
