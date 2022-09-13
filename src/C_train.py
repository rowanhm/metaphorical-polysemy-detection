from src.shared.global_variables import runtime
from src.training.worker import train_from_queue

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--runtime', type=float, default=runtime)
parser.add_argument('--exp', type=int, default=0)

args = parser.parse_args()

train_from_queue(experiment_id=args.exp, runtime=args.runtime)