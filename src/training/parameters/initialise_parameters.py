import itertools
import os
import random
import shutil
import pandas as pd

from src.shared.common import save_text, warn, info
from src.shared.global_variables import param_details_file, param_queue_file, results_file, experiment_dir, model_dir, \
    models, dropouts, n_layers, n_samples, seed, predictions_dir, alphas, hidden_sizes, learning_rate_divisors, \
    learning_rates
from src.training.parameters.parameters import ParameterKeys

random.seed(seed)


def initialise_params(experiment_id='test', force=False):

    experiment_dir_formatted = experiment_dir.format(experiment_id)
    if os.path.exists(experiment_dir_formatted):
        if force:
            info(f'Force deleting experiment {experiment_id} directory')
            shutil.rmtree(experiment_dir_formatted)
        else:
            warn(f'Experiment id {experiment_id} already exists; delete before proceeding')
            return

    os.makedirs(experiment_dir_formatted)
    os.makedirs(model_dir.format(experiment_id))
    os.makedirs(predictions_dir.format(experiment_id))

    all_params = [dropouts, n_layers, n_layers, hidden_sizes, hidden_sizes, learning_rates, learning_rate_divisors]
    grid = list(itertools.product(*all_params))
    samples = random.sample(grid, n_samples)

    parameter_dict = dict()
    param_index = 0
    for sample in samples:
        dropout, n_layers_1, n_layers_2, hidden_size_1, hidden_size_2, learning_rate, learning_rate_divisor = sample

        for model in models:

            alphas_filtered = alphas
            split = model.split('.')
            model_type = split[0]
            subtype = split[1]

            if model_type == 'wsd':
                alphas_filtered = [0]
            elif model_type == 'met' or (model_type == 'serial' and subtype == 'precomp'):
                alphas_filtered = [1]

            for alpha in alphas_filtered:
                parameter_dict[f'{model}.{param_index}'] = dict({
                    ParameterKeys.MODEL.name: model,
                    ParameterKeys.ALPHA.name: alpha,
                    ParameterKeys.DROPOUT.name: dropout,
                    ParameterKeys.N_LAYERS_1.name: n_layers_1,
                    ParameterKeys.N_LAYERS_2.name: n_layers_2,
                    ParameterKeys.HIDDEN_SIZE_1.name: hidden_size_1,
                    ParameterKeys.HIDDEN_SIZE_2.name: hidden_size_2,
                    # ParameterKeys.GNN_TYPE.name: gnn_type,
                    ParameterKeys.LEARNING_RATE.name: learning_rate,
                    ParameterKeys.LEARNING_RATE_DIVISOR.name: learning_rate_divisor
                })
                param_index += 1

    data = pd.DataFrame.from_dict(parameter_dict, orient='index')

    # Save data
    data.to_csv(param_details_file.format(experiment_id), index_label='param_id')
    save_text(param_queue_file.format(experiment_id), [i for i in parameter_dict.keys()])

    columns = ['param_id']
    for dataset in ['verbs', 'selection', 'random']:
        for result_type in ['accuracy', 'precision', 'recall', 'f1']:
            for thresholding in ['raw', 'single_thresh', 'multi_thresh']:
                columns += [f'{dataset}.{result_type}.{thresholding}']
        for datatype in ['ndcg', 'ranking_loss', 'ranking_precision', 'word_roc_auc', 'word_pr_auc', 'overall_roc_auc', 'overall_pr_auc']:
            columns += [f'{dataset}.{datatype}']

    for dataset in ['vuamc', 'semcor']:
        for result_type in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            for subset in ['train', 'dev', 'test']:
                columns += [f'{dataset}.{result_type}.{subset}']

    save_text(results_file.format(experiment_id), ['\t'.join(columns)])
    info(f'Built new experiment directory {experiment_id}')
    return experiment_id
