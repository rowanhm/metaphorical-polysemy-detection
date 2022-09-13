import argparse
import os.path
import random

import pandas as pd
import torch

from src.shared.common import info, open_pickle, get_file_list, open_csv_as_dict, save_text, save_csv
from src.shared.global_variables import sense_adjacency_file, sense_embs_file, word_embs_file, md_vuamc_dev_file, \
    train_batch_size, wsd_semcor_dev_file, wsd_semcor_train_dir, wsd_semcor_test_file, model_dir, device, \
    wsd_semcor_extracted_file, mpd_verbs_machine_file, mpd_mml_machine_file, mpd_oov_machine_file, \
    seed, experiment_dir
from src.training.utils.evaluation import evaluate
from src.training.utils.file_editing import FileEditor
from src.training.utils.model_initialiser import initialise_model
from src.training.utils.trainer_utils import make_static_train_loader

random.seed(seed)


def exemplify_melbert(experiment_id):


    info('Getting params of best MelBERT')
    file_editor = FileEditor(experiment_id)
    results = file_editor.get_results()

    best_param_id = ''
    best_dev = 0
    for index, row in results.iterrows():
        if index.startswith('met.sota'):
            if row['vuamc.f1.dev'] > best_dev:
                best_dev = row['vuamc.f1.dev']
                best_param_id = index

    params = file_editor.get_params_for_id(best_param_id)

    info(f'Initialising model {best_param_id}')
    token_emb_size = next(iter(make_static_train_loader(md_vuamc_dev_file, batch_size=train_batch_size,
                                                        shuffle=False)))[1].shape[1]
    word_emb_tensor = open_pickle(word_embs_file).cpu()
    sense_emb_tensor = open_pickle(sense_embs_file).cpu()
    adjacency_tensor = open_pickle(sense_adjacency_file).cpu()

    model = initialise_model(params=params, token_emb_size=token_emb_size, word_emb_tensor=word_emb_tensor,
                             sense_emb_tensor=sense_emb_tensor, contexts_dict=None,
                             sense_adjacency_tensor=adjacency_tensor)

    best_directory = model_dir.format(experiment_id)+f'{best_param_id}.pth'
    info(f'Recovering best from {best_directory}')
    model.load_state_dict(torch.load(best_directory, map_location=device))
    model.eval()

    # Open all the semcor data
    info('Preparing SemCor files')
    files = get_file_list(wsd_semcor_train_dir, end='.pkl')
    files += [wsd_semcor_dev_file, wsd_semcor_test_file]

    # files = files[:1]

    all_datapoint_ids = []
    all_predictions = []
    all_senses = []
    all_words = []

    info('Processing predictions')
    for i, file in enumerate(files):
        info(f'On file {os.path.basename(file)} ({i+1}/{len(files)})...')
        loader = make_static_train_loader(file, batch_size=train_batch_size, shuffle=False)

        with torch.no_grad():
            for (datapoint_ids, token_embs, sentence_embs, word_ids, wsd_opts_hot, wsd_opts_inds, mpd_opts_hot,
                 mpd_opts_inds, labels, precomp_wsd) in loader:
                log_probabilities = model.forward_metaphor(token_embs, sentence_embs, word_ids, wsd_opts_hot,
                                                           wsd_opts_inds, mpd_opts_hot, mpd_opts_inds, precomp_wsd)
                all_predictions += [round(probability) for probability in
                                    torch.exp(log_probabilities).tolist()]
                all_datapoint_ids += datapoint_ids
                all_senses += labels.tolist()
                all_words += word_ids.tolist()

    all_word_senses = [f'{w}:{s}' for w, s in zip(all_words, all_senses)]

    info('Converting to dictionary')
    sense_lookup = dict()
    for datapoint_id, metaphor, sense in zip(all_datapoint_ids, all_predictions, all_word_senses):
        if sense not in sense_lookup.keys():
            sense_lookup[sense] = ([metaphor], [datapoint_id])
        else:
            metaphors, datapoint_ids = sense_lookup[sense]
            sense_lookup[sense] = (metaphors + [metaphor], datapoint_ids + [datapoint_id])

    info('Making MPD predictions')
    # sense_lookup: "{word}:{synset}" -> (list(boolean metaphor), list(datapoint_ids))
    # vocab_dict = open_pickle(word_ids_file)
    # word_id_lookup = {}
    # for word, key in vocab_dict.items():
    #     assert key not in word_id_lookup.keys()
    #     word_id_lookup[key] = word
    #
    # sense_dict = open_pickle(sense_ids_file)
    # sense_id_lookup = {}
    # for sense, key in sense_dict.items():
    #     assert key not in sense_id_lookup.keys()
    #     sense_id_lookup[key] = sense

    eval_data = dict({
        'verbs': open_pickle(mpd_verbs_machine_file),
        'selection': open_pickle(mpd_mml_machine_file),
        'random': open_pickle(mpd_oov_machine_file)
    })
    results = dict()
    outputs = dict()
    for dataset, data in eval_data.items():
        (all_word_ids, all_datapoints, all_sense_ids, all_labels) = zip(*data)
        # Generate random probabilities
        all_mpd_probabilies = []
        for (word_id, sense_ids) in zip(all_word_ids, all_sense_ids):
            mpd_probs = []
            for sense_id in sense_ids:
                key = f'{word_id}:{sense_id}'
                if key in sense_lookup.keys():
                    predictions = sense_lookup[key][0]
                    if len(predictions) > 0:
                        pred = sum(predictions)/len(predictions)
                    else:
                        pred = random.uniform(0, 1)
                else:
                    pred = random.uniform(0, 1)
                mpd_probs += [pred]

            all_mpd_probabilies += [mpd_probs]
        result_dict, output = evaluate(all_datapoints, all_mpd_probabilies, all_labels)

        outputs[dataset] = output
        results[dataset] = result_dict

    info('Adding baseline output')
    info('Reformating')
    results_dict_melbert = dict()
    melbert_results_dataframe = pd.DataFrame.from_dict(results, orient='index')

    for dataset in melbert_results_dataframe.index:
        for data_type in melbert_results_dataframe.columns:
            results_dict_melbert[f'{dataset}.{data_type}'] = melbert_results_dataframe[data_type][dataset]
    # Adding stub
    for dataset in ['vuamc', 'semcor']:
        for result_type in ['loss', 'accuracy', 'precision', 'recall', 'f1']:
            for subset in ['train', 'dev', 'test']:
                results_dict_melbert[f'{dataset}.{result_type}.{subset}'] = None
    file_editor.save_result('melbert', results_dict_melbert)
    file_editor.save_predictions('melbert', outputs)

    info('Calculating % with inconsistency')
    min_occurences = 1
    total = 1
    prev_total = 0
    percent_output = []
    while total > 0:
        # Remove all senses with fewer that min_occurencens occurences
        total = 0
        erronous = 0
        for sense, (metaphors, datapoints) in sense_lookup.items():
            if len(datapoints) >= min_occurences:
                total += 1
                if 0 < sum(metaphors) < len(metaphors):  # not all =0 or all =1
                    erronous += 1

        if total > 0 and total != prev_total:
            info(f'For senses with >={min_occurences} occurences, {erronous}/{total} are erronous ({erronous*100/total}%)')
            percent_output.append({
                'min_occurences': min_occurences,
                'percent_erronous': erronous*100/total
            })
        min_occurences += 1
        prev_total = total
        if min_occurences > 20:
            break

    info('Saving percent')
    save_csv(experiment_dir.format(experiment_id)+'MelBERT_percents.tsv', percent_output)

    info('Saving examples')
    sentence_data = open_csv_as_dict(wsd_semcor_extracted_file, key_col='datapoint_id', val_col='sentence')
    sentence_data_2 = open_csv_as_dict(wsd_semcor_extracted_file, key_col='datapoint_id', val_col='word')
    sentence_data_3 = open_csv_as_dict(wsd_semcor_extracted_file, key_col='datapoint_id', val_col='sense')

    output = ["word\tsense\tmetaphoricity_prediction\tsentence"]
    for j, (sense, (metaphors, datapoints)) in enumerate(sense_lookup.items()):
        if 0 < sum(metaphors) < len(metaphors):  # not all =0 or all =1
            for i, (metaphor, datapoint_id) in enumerate(zip(metaphors, datapoints)):
                # It is erronous
                datapoint_id_first = ".".join(datapoint_id.split('.')[:2] + ['0'])
                sentence = sentence_data[datapoint_id_first]
                word = sentence_data_2[datapoint_id]
                sense = sentence_data_3[datapoint_id]
                output += [f'{word}\t{sense}\t{metaphor}\t{sentence}']

    save_text(experiment_dir.format(experiment_id)+'MelBERT_contradictions.tsv', output)
    info('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    args = parser.parse_args()

    experiment = args.exp
    exemplify_melbert(experiment)