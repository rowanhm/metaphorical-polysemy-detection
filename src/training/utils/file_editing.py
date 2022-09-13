from filelock import FileLock
import os
import pandas as pd
import torch

from src.training.utils.model_initialiser import initialise_model
from src.shared.common import read_text, save_text, save_list_csv
from src.shared.global_variables import param_details_file, param_queue_file, results_file, model_dir, seed, \
    predictions_dir

torch.manual_seed(seed)


class FileEditor:

    def __init__(self, experiment_id):

        self.param_details = pd.read_csv(param_details_file.format(experiment_id), index_col='param_id')

        self.param_queue_file = param_queue_file.format(experiment_id)
        self.results_file = results_file.format(experiment_id)
        self.model_dir = model_dir.format(experiment_id)
        self.predictions_dir = predictions_dir.format(experiment_id)

    def get_params(self):
        return self.param_details

    def get_params_for_id(self, param_id):
        parad_id_details = self.param_details.loc[param_id].to_dict()
        return parad_id_details

    def pop_param(self):
        param_id = None
        with FileLock(self.param_queue_file + ".lock", timeout=-1):
            queue = read_text(self.param_queue_file)
            if queue:
                param_id = queue.pop()
                save_text(self.param_queue_file, queue)
        return param_id

    def queue_param(self, param_id):
        with FileLock(self.param_queue_file + ".lock", timeout=-1):
            queue = read_text(self.param_queue_file)
            queue += [param_id]
            save_text(self.param_queue_file, queue)

    def get_results(self):
        results = pd.read_csv(self.results_file, index_col='param_id', sep='\t')
        return results

    def save_result(self, param_id, result):
        assert result is not None
        result_wrapped = dict({
            param_id: result
        })
        new_result = pd.DataFrame.from_dict(result_wrapped, orient='index')

        with FileLock(self.results_file + ".lock", timeout=-1):
            results = pd.read_csv(self.results_file, index_col='param_id', sep='\t')
            results = results.append(new_result)
            results.index.name = 'param_id'
            results.to_csv(self.results_file, sep='\t')

    def save_model(self, param_id, model):
        torch.save(model.state_dict(), os.path.join(self.model_dir, f'{param_id}.pth'))

    def save_predictions(self, param_id, predictions_dict):
        # dataset -> list of zipped datapoints
        for dataset, data in predictions_dict.items():
            if data:
                save_list_csv(os.path.join(self.predictions_dir, f'{param_id}.{dataset}.tsv'), data)

    def get_predictions(self, param_id, dataset):
        predictions = pd.read_csv(os.path.join(self.predictions_dir, f'{param_id}.{dataset}.tsv'),
                                  index_col='datapoint_id', sep='\t')
        return predictions

    def load_model(self, param_id, token_emb_size, word_emb_tensor, sense_emb_tensor, contexts_dict):
        params = self.get_params_for_id(param_id)
        model = initialise_model(params, token_emb_size, word_emb_tensor, sense_emb_tensor, contexts_dict)
        model.load_state_dict(torch.load(os.path.join(self.model_dir, f'{param_id}.pth')))
        return model
