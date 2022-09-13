import torch

from src.shared.common import open_pickle
from src.shared.global_variables import device, seed, sense_ids_file

torch.manual_seed(seed)

# TODO get rid of this if possible, it is quite ugly
sense_inventory_size = len(open_pickle(sense_ids_file)) + 1


def reformat_data(sentences_data):
    # Flatten data in form [(sentence_emb, [(context_emb, word_id, pos_id, synset_options, label), ...]), ...]
    reformatted_data = []
    for (sentence_emb, annotations) in sentences_data:
        for (datapoint_id, token_emb, word_id, pos_id, synset_wsd_options, synset_mpd_options, label, precomputed_wsd) in annotations:
            reformatted_data += [
                (datapoint_id, token_emb.cpu(), sentence_emb.cpu(), word_id, pos_id, synset_wsd_options,
                 synset_mpd_options, label, precomputed_wsd)]
    return reformatted_data


# def data_reformat_generator(sentences_data):
#     buffer = []
#     sentences_data_index = 0
#     while not (not buffer and sentences_data_index >= len(sentences_data)):  # While it is not the case that both are empty
#         # Refresh the buffer
#         if not buffer:
#             (sentence_emb, annotations) = sentences_data[sentences_data_index]
#             for (token_emb, word_id, pos_id, synset_options, label) in annotations:
#                 buffer += [(token_emb, sentence_emb, word_id, pos_id, synset_options, label)]
#             sentences_data_index += 1
#         yield buffer.pop(0)  # NB this will crash if annotations was an empty list

def convert_sense_list(sense_options_list):
    max_length = max([len(options) for options in sense_options_list])
    batch_size = len(sense_options_list)
    sense_options_hot = torch.zeros([batch_size, sense_inventory_size]).bool()
    sense_options_indices = torch.zeros([batch_size, max_length]).long()
    for i, synset_options in enumerate(sense_options_list):
        sense_options_hot[i, synset_options] = 1
        sense_options_indices[i, :len(synset_options)] = torch.tensor(synset_options)
    return sense_options_hot.to(device=device), sense_options_indices.to(device=device)


def train_collate_fn(data):
    (datapoint_ids, token_embs, sentence_embs, word_ids, pos_ids, synset_wsd_options_list, synset_mpd_options_list,
     labels, precomputed_wsd) = zip(*data)

    token_embs = torch.stack(token_embs).to(device=device)
    sentence_embs = torch.stack(sentence_embs).to(device=device)
    word_ids = torch.tensor(word_ids).to(device=device)
    labels = torch.tensor(labels).to(device=device)

    # Synset options is list of lists
    mpd_options_hot, mpd_options_indices = convert_sense_list(synset_mpd_options_list)
    wsd_options_hot, wsd_options_indices = convert_sense_list(synset_wsd_options_list)

    return datapoint_ids, token_embs, sentence_embs, word_ids, wsd_options_hot, wsd_options_indices, mpd_options_hot, \
           mpd_options_indices, labels, precomputed_wsd


# def average_datasets_results(results_dataframe, vuamc_fraction):
#     # assert sorted([d for d in ratio_dict.keys()]) == sorted(results_dataframe.index)
#
#     assert 0 <= vuamc_fraction
#     assert 1 >= vuamc_fraction
#
#     output = dict()
#     for result_type in results_dataframe.columns:
#         result = 0
#         for dataset in results_dataframe.index:
#             if dataset == 'vuamc':
#                 probability = vuamc_fraction
#             else:
#                 assert dataset == 'semcor'
#                 probability = 1 - vuamc_fraction
#             if probability > 0:  # Avoid the computation as otherwise it will be null values
#                 result += (probability * results_dataframe[result_type][dataset])
#
#         output[result_type] = result
#     return output


def improved(best_so_far, new_result, metric):
    if metric == 'loss':
        return new_result < best_so_far
    else:
        assert metric == 'f1'
        return new_result > best_so_far


def make_static_train_loader(file_dir, batch_size, shuffle, limit_data=None):
    data = open_pickle(file_dir)
    data = reformat_data(data)[:limit_data]
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, collate_fn=train_collate_fn,
                                         shuffle=shuffle)
    return loader


def make_static_eval_loader(file_dir, batch_size, limit_data=None):
    data = open_pickle(file_dir)[:limit_data]
    loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, collate_fn=eval_collate_fn,
                                         shuffle=False)
    return loader


def eval_collate_fn(data):
    (word_ids, datapoint_ids, synset_options_list, metaphor_labels) = zip(
        *data)  # list, list of lists, list of lists, list of lists

    try:
        word_ids = torch.tensor(word_ids).to(device=device)
    except ValueError:
        print('')

    mpd_options_hot, mpd_options_indices = convert_sense_list(synset_options_list)

    # assert set(flatten(synset_options_list)) == set(torch.nonzero(sense_options_hot, as_tuple=True)[1].tolist())
    return word_ids, datapoint_ids, mpd_options_hot, mpd_options_indices, metaphor_labels
