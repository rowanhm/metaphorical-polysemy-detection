import argparse

from src.shared.common import info
from src.training.parameters.initialise_parameters import initialise_params

parser = argparse.ArgumentParser()
parser.add_argument('--force', dest='force', action='store_true')
parser.add_argument('--exp', type=int, default=0)
parser.set_defaults(force=False)
args = parser.parse_args()

info('Building params and queue')
initialise_params(experiment_id=args.exp, force=args.force)
