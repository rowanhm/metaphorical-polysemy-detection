from src.postprocessing.baselines import random_baseline
from src.postprocessing.melbert_examples import exemplify_melbert
from src.postprocessing.result_extractor import extract_results

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=int, default=0)
args = parser.parse_args()

experiment = args.exp

exemplify_melbert(experiment)
random_baseline(experiment)
extract_results(experiment)
