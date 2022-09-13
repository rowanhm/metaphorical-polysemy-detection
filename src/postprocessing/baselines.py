import random
import pandas as pd

from src.shared.common import open_pickle, info
from src.shared.global_variables import mpd_verbs_machine_file, mpd_mml_machine_file, mpd_oov_machine_file, seed
from src.training.utils.evaluation import evaluate
from src.training.utils.file_editing import FileEditor

random.seed(seed)


def random_baseline(experiment_id):
    # Load the evaluation data
    eval_data = dict({
        'verbs': open_pickle(mpd_verbs_machine_file),
        'selection': open_pickle(mpd_mml_machine_file),
        'random': open_pickle(mpd_oov_machine_file)
    })

    info('Generating random baseline data and evaluating')
    random_outputs = dict()
    majority_outputs = dict()
    random_results = dict()
    majority_results = dict()

    for dataset, data in eval_data.items():

        (_, all_datapoints, _, all_labels) = zip(*data)
        # Generate random probabilities
        all_random_probabilities = []
        all_majority_probabilities = []
        for datapoints in all_datapoints:
            random_probs = []
            for _ in datapoints:
                random_probs += [random.uniform(0, 1)]
            all_random_probabilities += [random_probs]
            all_majority_probabilities += [[0.0] * len(datapoints)]
        random_result_dict, random_output = evaluate(all_datapoints, all_random_probabilities, all_labels)
        majority_result_dict, majority_output = evaluate(all_datapoints, all_majority_probabilities, all_labels)

        random_outputs[dataset] = random_output
        majority_outputs[dataset] = majority_output

        random_results[dataset] = random_result_dict
        majority_results[dataset] = majority_result_dict

    info('Reformating')
    results_dict_random = dict()
    results_dict_majority = dict()
    random_results_dataframe = pd.DataFrame.from_dict(random_results, orient='index')
    majority_results_dataframe = pd.DataFrame.from_dict(majority_results, orient='index')

    for dataset in random_results_dataframe.index:
        for data_type in random_results_dataframe.columns:
            results_dict_random[f'{dataset}.{data_type}'] = random_results_dataframe[data_type][dataset]
            results_dict_majority[f'{dataset}.{data_type}'] = majority_results_dataframe[data_type][dataset]
    # Adding stub
    for dataset in ['vuamc', 'semcor']:
        for result_type in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            for subset in ['train', 'dev', 'test']:
                results_dict_random[f'{dataset}.{result_type}.{subset}'] = None
                results_dict_majority[f'{dataset}.{result_type}.{subset}'] = None

    info('Saving to result file')
    file_editor = FileEditor(experiment_id)
    file_editor.save_result('random', results_dict_random)
    file_editor.save_predictions('random', random_outputs)
    file_editor.save_result('majority', results_dict_majority)
    file_editor.save_predictions('majority', majority_outputs)


if __name__ == "__main__":
    random_baseline(0)
