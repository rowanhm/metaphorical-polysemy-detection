import torch
from nltk.corpus import wordnet as wn

# testing
test_mode = False

# PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories
root = './'
raw_data_dir = root + 'data/raw/'
extracted_data_dir = root + 'data/extracted/'
machine_data_dir = root + 'data/machine/'
output_dir = root + 'output/'

sense_ids_file = machine_data_dir + 'sense_ids.pkl'
sense_embs_file = machine_data_dir + 'sense_embs.pkl'
sense_adjacency_file = machine_data_dir + 'sense_adjacency.pkl'

word_ids_file = machine_data_dir + 'word_ids.pkl'
word_embs_file = machine_data_dir + 'word_embs.pkl'

concreteness_raw_file = raw_data_dir + 'brysbaert_concreteness.csv'
concreteness_extracted_file = extracted_data_dir + 'concreteness.tsv'
concreteness_machine_file = machine_data_dir + 'concreteness.pkl'

## MD

md_vuamc_raw_file = raw_data_dir + 'emnlp2018-novel-metaphors/annotation/VUAMC_with_novelty_scores.xml'
md_vuamc_extracted_file = extracted_data_dir + 'md_vuamc.tsv'
md_vuamc_vocab_file = extracted_data_dir + 'md_vuamc_vocab.pkl'
md_vuamc_train_dir = machine_data_dir + 'md_vuamc_train/'
md_vuamc_dev_file = machine_data_dir + 'md_vuamc_dev.pkl'
md_vuamc_test_file = machine_data_dir + 'md_vuamc_test.pkl'

md_vuamc_test_subset_file = machine_data_dir + 'md_vuamc_test_subset.pkl'

md_vuamc_xml_file = extracted_data_dir + 'md_vuamc.data.xml'
md_vuamc_wsd_file = machine_data_dir + 'md_vuamc_predictions.pkl'

## WSD

wsd_semcor_extracted_file = extracted_data_dir + 'wsd_semcor.tsv'
wsd_semcor_vocab_file = extracted_data_dir + 'wsd_semcor_vocab.pkl'
wsd_semcor_train_dir = machine_data_dir + 'wsd_semcor_train/'
wsd_semcor_dev_file = machine_data_dir + 'wsd_semcor_dev.pkl'
wsd_semcor_test_file = machine_data_dir + 'wsd_semcor_test.pkl'

## MPD

mpd_labelled_files_dir = raw_data_dir + 'mpd_labelled/'
mpd_machine_dir = machine_data_dir + 'mpd/'

mpd_mml_vocab_file = raw_data_dir + 'mpd_mml_vocab.pkl'
mpd_mml_raw_file = raw_data_dir + 'mpd_mml_word_list.txt'
mpd_mml_csv_file = mpd_labelled_files_dir + 'mml.tsv'
mpd_mml_machine_file = mpd_machine_dir + 'mml.pkl'

mpd_multi_info = raw_data_dir + 'multi_wn_info.tsv'
mpd_multi_raw_dir = raw_data_dir + 'wns/'
mpd_multi_vocab_file = extracted_data_dir + 'mpd_multi_vocab.pkl'
mpd_multi_csv_file = mpd_labelled_files_dir + 'multi_{}.tsv'
mpd_multi_machine_file = mpd_machine_dir + 'multi_{}.pkl'

mpd_oov_csv_file = mpd_labelled_files_dir + 'oov.tsv'
mpd_oov_machine_file = mpd_machine_dir + 'oov.pkl'

mpd_verbs_raw_file = raw_data_dir + 'Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt'
mpd_verbs_csv_file = mpd_labelled_files_dir + 'verbs.tsv'
mpd_verbs_machine_file = mpd_machine_dir + 'verbs.pkl'

mpd_all_csv_file = extracted_data_dir + 'mpd_all_unlabelled.tsv'
mpd_all_machine_file = machine_data_dir + 'mpd_all_unlabelled.pkl'

# NB the below require initialisation
experiment_dir = output_dir + 'experiment_{}/'
param_details_file = experiment_dir + 'parameters.csv'
param_queue_file = experiment_dir + 'queue.txt'
results_file = experiment_dir + 'results.tsv'
model_dir = experiment_dir + 'models/'
predictions_dir = experiment_dir + 'predictions/'

num_words_in_eval_sets = 100

wn_pos_reverse = dict({
    'v': 'verb',
    'n': 'noun',
    'r': 'adv',
    'a': 'adj',
    's': 'adj'
})

wn_pos = dict({
    'verb': [wn.VERB],
    'noun': [wn.NOUN],
    'adj': [wn.ADJ, wn.ADJ_SAT],
    'adv': [wn.ADV],
})

pos_2_id = dict({
    'noun': 0,
    'verb': 1,
    'adj': 2,
    'adv': 4
})

# Hardcoded params
embedding_mode = 'take-last'  # or 'average-4'
combo_mode = 'take-first'  # or 'average'
bert_model = 'bert-base-cased'  # large doesn't work with EWISER as the embs are diff dims

preprocessing_batch_size = 16
train_batch_size = 128
stopping_iterations = 5
eval_steps = 50
runtime = 1.9  # hours
n_samples = 20

# Tunable parameters
alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]
models = ['met.baseline', 'met.sota', 'serial.baseline.baseline', 'serial.sota.baseline']
dropouts = [0.1, 0.2, 0.3, 0.4]
n_layers = [1, 2, 3, 4]
hidden_sizes = [100, 300, 500]
learning_rates = [0.005, 0.001, 0.0005, 0.0001]
learning_rate_divisors = [1, 10]
seed = 42
