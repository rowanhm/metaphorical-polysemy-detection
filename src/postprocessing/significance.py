import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.shared.global_variables import mpd_verbs_csv_file, mpd_oov_csv_file, mpd_mml_csv_file, \
    wsd_semcor_extracted_file, md_vuamc_extracted_file, seed, sense_ids_file
from src.training.utils.file_editing import FileEditor
from src.shared.common import open_pickle, info

np.random.seed(seed)


def significance(experiment_id, param_id_1, param_id_2, metric, r=1000, expected_difference=None):

    setting_datafiles = dict({
        'selection': mpd_mml_csv_file,
        'random': mpd_oov_csv_file,
        'verbs': mpd_verbs_csv_file
    })

    reshape_predictions = False
    file_editor = FileEditor(experiment_id)
    if metric[:3] == 'mpd':
        _, dataset, metric_name = metric.split('.')
        param_1_preds = file_editor.get_predictions(param_id_1, dataset)
        param_2_preds = file_editor.get_predictions(param_id_2, dataset)
        param_1_preds.rename(columns={"p_met": "param_1_preds"}, inplace=True)
        param_2_preds.rename(columns={"p_met": "param_2_preds"}, inplace=True)
        datafile = setting_datafiles[dataset]
        if metric_name == 'f1':
            eval_fn = lambda pred, truth: f1_score(y_true=truth, y_pred=[round(p) for p in pred], average='binary')
        else:
            eval_fn = lambda pred, truth: word_roc_auc(pred, truth)
            reshape_predictions = True

    else:
        if metric == 'wsd':
            param_1_preds = file_editor.get_predictions(param_id_1, 'semcor')
            param_2_preds = file_editor.get_predictions(param_id_2, 'semcor')
            datafile = wsd_semcor_extracted_file
            eval_fn = lambda pred, truth: f1_score(y_true=truth, y_pred=pred, average='micro')
        else:
            assert metric == 'md'
            param_1_preds = file_editor.get_predictions(param_id_1, 'vuamc')
            param_2_preds = file_editor.get_predictions(param_id_2, 'vuamc')
            datafile = md_vuamc_extracted_file
            eval_fn = lambda pred, truth: f1_score(y_true=truth, y_pred=pred, average='binary')
        param_1_preds.rename(columns={"predictions": "param_1_preds"}, inplace=True)
        param_2_preds.rename(columns={"predictions": "param_2_preds"}, inplace=True)

    predictions = pd.merge(param_1_preds, param_2_preds, on="datapoint_id")
    datapoints = pd.read_csv(datafile, index_col='datapoint_id', sep='\t')
    predictions = pd.merge(datapoints, predictions, how="right", on="datapoint_id")

    if metric[:3] == 'mpd' or metric == 'md':
        # predictions.rename(columns={"metaphor": "truth"}, inplace=True)
        predictions["truth"] = [int(p) for p in predictions['metaphor']]
    else:
        assert metric == 'wsd'
        # If WSD, lookup the sense
        wsd_vocab = open_pickle(sense_ids_file)
        predictions['truth'] = [wsd_vocab[sense_id] for sense_id in predictions['sense'].values]

    observed_diff = evaluate_diff(predictions, eval_fn, reshape=reshape_predictions)
    if expected_difference is not None:
        assert expected_difference == observed_diff
        # f'Expected difference = {expected_difference}, observed difference = {observed_difference} ({v1_pred}/{v1}; {v2_pred}/{v2})')

    predictions_swapped = predictions.copy()
    predictions_swapped.rename(columns={"param_1_preds": "temp"}, inplace=True)
    predictions_swapped.rename(columns={"param_2_preds": "param_1_preds"}, inplace=True)
    predictions_swapped.rename(columns={"temp": "param_2_preds"}, inplace=True)

    s = 0  # s is number of times the difference is greater that observed

    for i in range(r):
        if i % 200 == 0:
            info(f'On iteration {i}/{r}')

        choice = np.random.choice([True, False], size=len(predictions))
        shuffled = pd.concat((predictions[choice], predictions_swapped[~choice]))

        shuffled_diff = evaluate_diff(shuffled, eval_fn, reshape=reshape_predictions)
        if shuffled_diff >= observed_diff:
            s += 1

    p = (s+1) / (r+1)
    return p


def word_roc_auc(input, truth):
    word_roc_aucs = []
    for (labels, probabilities) in zip(truth, input):
        if not (all([l == 1 for l in labels]) or all([l == 0 for l in labels])):
            word_roc_aucs += [roc_auc_score(labels, probabilities)]
    word_roc_auc = np.mean(word_roc_aucs)
    return word_roc_auc


def evaluate_diff(predictions, eval_fn, reshape=False):

    if not reshape:
        data_1 = predictions['param_1_preds']
        data_2 = predictions['param_2_preds']
        truth = predictions['truth']
    else:
        # for word roc-auc, regroup these predictions by word
        grouped_preds = predictions.groupby("word")
        data_1 = grouped_preds['param_1_preds'].apply(list).values
        data_2 = grouped_preds['param_2_preds'].apply(list).values
        truth = grouped_preds['truth'].apply(list).values

    result_1 = eval_fn(data_1, truth)
    result_2 = eval_fn(data_2, truth)
    diff = result_1 - result_2
    return diff