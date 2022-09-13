import argparse

from src.shared.common import info
from src.preprocessing.Aa_build_sense_tensor import build_sense_tensor
from src.preprocessing.Ab_build_sense_adjacency import build_adjacency_matrix
from src.preprocessing.Ba_csvify_vuamc import extract_vuamc
from src.preprocessing.Bb_csvify_semcor import extract_semcor
from src.preprocessing.C_build_word_tensor import build_word_tensor
from src.preprocessing.D_machinify_training_csvs import build_train_machine_data
from src.preprocessing.Ea_csvify_wordnet_verbs import build_verb_test_data
from src.preprocessing.Eb_csvify_wordnet_all import build_final_eval_data
from src.preprocessing.F_machinify_wordnet_csvs import build_machine_eval_data

parser = argparse.ArgumentParser()
parser.add_argument('--conventional', dest='conv', action='store_true')
parser.set_defaults(conv=False)
args = parser.parse_args()

info('===A.a===')
build_sense_tensor()

info('===A.b===')
build_adjacency_matrix()

info('===B.a===')
extract_vuamc(args.conv)

info('===B.b===')
extract_semcor()

info('===C===')
build_word_tensor()

info('===D===')
build_train_machine_data()

info('===E.a===')
build_verb_test_data()

info('===E.b===')
build_final_eval_data()

info('===F===')
build_machine_eval_data()
