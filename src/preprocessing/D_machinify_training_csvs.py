import os
import shutil
from math import ceil

import torch
from transformers import BertTokenizerFast, BertModel
import random

from src.shared.common import open_dict_csv, chunks, open_pickle, info, save_pickle, flatten, warn
from src.shared.global_variables import device, pos_2_id, wn_pos, bert_model, embedding_mode, combo_mode, \
    preprocessing_batch_size, sense_ids_file, word_ids_file, wsd_semcor_extracted_file, \
    md_vuamc_extracted_file, wsd_semcor_train_dir, wsd_semcor_dev_file, wsd_semcor_test_file, md_vuamc_train_dir, \
    md_vuamc_dev_file, \
    md_vuamc_test_file, test_mode, seed
from src.preprocessing.helper import get_synset_names

torch.manual_seed(seed)
random.seed(seed)


def save_data_folders(directory, data, n_data_points=1024):

    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        # Empty directory
        for root, dirs, files in os.walk(directory):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    for i, datasubset in enumerate(chunks(data, n=n_data_points)):
        save_pickle(os.path.join(directory, f'{i}.pkl'), datasubset)


def build_train_machine_data(testing=False):

    # --------------------------------------
    info("Loading dictionaries")
    sense_2_id = open_pickle(sense_ids_file)
    word_2_id = open_pickle(word_ids_file)

    # --------------------------------------
    info(f"Initialising BERT using device {device}")
    tokenizer = BertTokenizerFast.from_pretrained(bert_model)
    bert = BertModel.from_pretrained(bert_model).to(device=device)
    bert.eval()
    torch.no_grad()

    for (filename, mode, file_in, train_out, dev_out, test_out, wsd_predictions) in \
            [('vuamc', 'metaphor', md_vuamc_extracted_file, md_vuamc_train_dir, md_vuamc_dev_file, md_vuamc_test_file, None),
             ('semcor', 'sense', wsd_semcor_extracted_file, wsd_semcor_train_dir, wsd_semcor_dev_file, wsd_semcor_test_file, None)]:

        info(f"Processing {filename}")

        # --------------------------------------
        info("Opening data and collating")
        data = open_dict_csv(file_in)

        data_lookup = dict()
        sentence_rolling = []
        data_bundled = []
        prev_sentence_id = ''
        sentence = ''

        # Build datapoint_id -> WSD predictions map
        # if mode == 'metaphor':
        #     assert wsd_predictions is not None
        #     wsd_pred_map = {}
        #     wsd_predictions = open_pickle(wsd_predictions)
        #     datapoints_without_pred = set()
        #     for datapoint in data:
        #         datapoint_id = datapoint['datapoint_id']
        #         wsd_id = datapoint['wsd_index']
        #         if wsd_id in wsd_predictions.keys():
        #             wsd_pred_map[datapoint_id] = wsd_predictions[wsd_id]
        #         else:
        #             warn(f'No WSD predictions for datapoint {datapoint_id}')
        #             datapoints_without_pred.add(datapoint_id)

        for datapoint in data:
            datapoint_id = datapoint['datapoint_id']
            assert datapoint_id not in data_lookup.keys()
            data_lookup[datapoint_id] = datapoint

            _, sentence_id, token_id = datapoint_id.split('.')
            sentence_id = int(sentence_id)
            token_id = int(token_id)

            if sentence_id == prev_sentence_id:
                assert datapoint['sentence'] == ''
                assert token_id == prev_token_id + 1
            else:
                end_offset = 0
                data_bundled += [(sentence_id, sentence, sentence_rolling)]
                sentence_rolling = []
                prev_sentence_id = sentence_id
                sentence = datapoint['sentence'].replace('\\', '')

            start_offset = int(datapoint['start_offset'])
            assert start_offset >= end_offset  # Word later in sentence than previous
            end_offset = int(datapoint['end_offset'])
            sentence_rolling += [(datapoint_id, (start_offset, end_offset))]
            prev_token_id = token_id
        # Add final sentence
        data_bundled += [(sentence_id, sentence, sentence_rolling)]
        data_bundled = data_bundled[1:]  # Remove initial stub datapoint

        if testing:
            data_bundled = data_bundled[:10]

        # --------------------------------------
        info("Aligning tokens with BERT scheme")
        included = 0
        skipped_nt = 0
        skipped_other = 0
        data_bert_tokens = []
        for (sentence_id, sentence, annotation) in data_bundled:

            token_data = tokenizer(sentence, return_offsets_mapping=True)
            bert_tokens = token_data.data['input_ids']
            bert_token_offsets = token_data.data['offset_mapping']

            bert_token_offsets_numbered = list(enumerate(bert_token_offsets))[1:-1]  # Exclude first/last tok offsets

            annotation_bert = []
            for idx, (start_offset, end_offset) in annotation:

                bert_start = -1
                bert_end = -1

                # Find start
                for token_x, (bert_start_offset, _) in bert_token_offsets_numbered:
                    if bert_start_offset < start_offset:
                        continue

                    if bert_start_offset == start_offset:
                        bert_start = token_x
                        break

                # Find end
                for token_y, (_, bert_end_offset) in sorted(bert_token_offsets_numbered, reverse=True):
                    if bert_end_offset > end_offset:
                        continue

                    if bert_end_offset == end_offset:
                        bert_end = token_y
                        break

                if bert_start == -1 or bert_end == -1:
                    if len(sentence) >= end_offset+3:
                        if sentence[end_offset:end_offset+3] != "n't":
                            skipped_other += 1
                        else:
                            skipped_nt += 1
                    else:
                        skipped_other += 1
                else:
                    included += 1
                    annotation_bert += [(idx, (bert_start, bert_end))]

            if annotation_bert:
                data_bert_tokens += [(sentence_id, bert_tokens, annotation_bert)]

        info(f"{included} included ({skipped_nt} excluded for n't, {skipped_other} others)")

        # --------------------------------------
        info("Computing BERT embeddings")
        data_bert_embeddings = []
        batches = chunks(data_bert_tokens, preprocessing_batch_size)
        for batch_id, batch in enumerate(batches):

            if (batch_id+1) % 50 == 0:
                info(f"On batch {batch_id + 1}/{ceil(len(data_bert_tokens) / preprocessing_batch_size)}")

            # Build input tensor
            sentence_ids, bert_tokens, bert_annos = zip(*batch)

            tensor_shape = [len(bert_tokens), max([len(tokens) for tokens in bert_tokens])]
            tokens_tensor = torch.zeros(tensor_shape).long().to(device=device)
            attention_tensor = torch.zeros(tensor_shape).bool().to(device=device)

            token_lengths = []
            for i, tokens in enumerate(bert_tokens):
                length = len(tokens)

                tokens_tensor[i, :length] = torch.tensor(tokens)
                attention_tensor[i, :length] = torch.tensor([1] * length)

                token_lengths += [length]

            output = bert(tokens_tensor, attention_mask=attention_tensor, output_hidden_states=True)[2]

            if embedding_mode == 'take-last':
                bert_embs_batch = output[-1]
            else:
                assert embedding_mode == 'average-4'
                bert_embs_batch = output[-1] + output[-2] + output[-3] + output[-4]
                bert_embs_batch = torch.div(bert_embs_batch, 4)

            for i, (sentence_id, bert_anno, length) in enumerate(zip(sentence_ids, bert_annos, token_lengths)):

                bert_embs_sentence = bert_embs_batch[i, :length]
                data_bert_embeddings += [(sentence_id, bert_embs_sentence.detach().cpu(), bert_anno)]

        # --------------------------------------
        info("Extracting relevant BERT embeddings")
        data_dict = dict()

        for sentence_id, bert_embs, bert_anno in data_bert_embeddings:
            sentence_emb = bert_embs[0]

            word_emb_dict = dict()
            for idx, (first_token, last_token) in bert_anno:

                if combo_mode == 'take-first':
                    token_emb = bert_embs[first_token]
                else:
                    assert combo_mode == 'average'
                    token_emb = torch.mean(bert_embs[first_token:last_token+1], dim=0)

                word_emb_dict[idx] = token_emb

            data_dict[sentence_id] = (sentence_emb, word_emb_dict)

        # --------------------------------------
        info("Building datafile and saving")

        skipped_singleton_synset = 0
        final_datapoints = 0

        machine_data = []  # (word_emb, sentence_emb, synset_options_ids, label)
        excluded_wsd_pred = 0
        for sentence_id, (sentence_emb, word_emb_dict) in data_dict.items():

            sentence_data = []

            for (datapoint_id, contextual_word_emb) in word_emb_dict.items():

                datapoint = data_lookup[datapoint_id]

                lemma = datapoint['lemma']
                word = datapoint['word']
                pos = datapoint['pos']

                lemma_synsets_wsd = get_synset_names(lemma, wn_pos[pos])
                word_synsets_wsd = get_synset_names(word, wn_pos[pos])
                synsets_wsd = set(lemma_synsets_wsd).union(set(word_synsets_wsd))

                synsets_mpd = set(get_synset_names(lemma)).union(set(get_synset_names(word)))

                # Make synset options ids
                synsets_wsd = [sense_2_id[synset] for synset in list(synsets_wsd)]
                synsets_mpd = [sense_2_id[synset] for synset in list(synsets_mpd)]

                assert len(synsets_mpd) >= len(synsets_wsd)
                assert set(synsets_wsd) <= set(synsets_mpd)

                if mode == 'sense':
                    label = sense_2_id[datapoint['sense']]
                    assert label in synsets_wsd and label in synsets_mpd
                    assert len(set(synsets_mpd)) == len(synsets_mpd) and len(set(synsets_wsd)) == len(synsets_wsd)# Each synset unique
                    assert len(synsets_wsd) > 1

                    # Convert label to index
                    # label = synsets.index(label)

                    wsd_preds = None
                else:
                    assert mode == 'metaphor'
                    assert len(synsets_wsd) >= 1
                    label = int(datapoint['metaphor'])

                    # if datapoint_id not in wsd_pred_map.keys():
                    #     assert datapoint_id in datapoints_without_pred
                    #     continue

                    wsd_preds_raw = None #wsd_pred_map[datapoint_id] #TODO This can be removed
                    wsd_preds_indices = []
                    wsd_preds_probabilities = []
                    # exclude = False
                    # renormalise = False
                    # for synset_offset, probability in wsd_preds_raw:
                    #     synset_name = wn.synset_from_pos_and_offset(synset_offset[-1], int(synset_offset[:-1])).name()
                    #     if synset_name not in sense_2_id.keys():
                    #         # renormalise = True
                    #         continue
                    #     synset_index = sense_2_id[synset_name]
                    #     if synset_index not in synsets_mpd:
                    #         excluded_wsd_pred += 1
                    #         exclude = True
                    #         continue
                    #     wsd_preds_indices.append(synset_index)
                    #     wsd_preds_probabilities.append(probability)
                    # assert len(wsd_preds_indices) == len(wsd_preds_probabilities)
                    # if (not exclude) and len(wsd_preds_indices) == 0:
                    #     exclude = True
                    #     excluded_wsd_pred += 1
                    # if exclude:
                    #     continue
                    # total = sum(wsd_preds_probabilities)
                    # wsd_preds_probabilities = [w/total for w in wsd_preds_probabilities]
                    # assert sum(wsd_preds_probabilities) < 1.0
                    # wsd_preds = (wsd_preds_indices, wsd_preds_probabilities)
                sentence_data += [(datapoint_id, contextual_word_emb, word_2_id[word], pos_2_id[pos], synsets_wsd, synsets_mpd, label, None)]

            if len(sentence_data) != 0:
                final_datapoints += len(sentence_data)
                machine_data += [(sentence_emb, sentence_data)]

        info(f"{skipped_singleton_synset} datapoints skipped because no choice and {excluded_wsd_pred} because of WSD gap, leaving final total of {final_datapoints} datapoints")

        del data_bundled
        del data
        del data_dict
        del data_bert_embeddings
        del data_bert_tokens

        info("Datasplits (8:1:1)")
        info("Datasplits (8:1:1)")
        random.shuffle(machine_data)

        num_datapoints = len(machine_data)
        eighth = round(num_datapoints*0.8)
        ninth = round(num_datapoints*0.9)

        # Compute final distribution
        sentences_train = 0
        sentences_dev = 0
        sentences_test = 0

        annotations_train = 0
        annotations_dev = 0
        annotations_test = 0

        for i, (_, sentence_data) in enumerate(machine_data):
            if i < eighth:
                # Train
                sentences_train += 1
                annotations_train += len(sentence_data)
            elif i >= ninth:
                # Test
                sentences_test += 1
                annotations_test += len(sentence_data)
            else:
                # Dev
                sentences_dev += 1
                annotations_dev += len(sentence_data)

        info(f'Sentence count {sentences_train}/{sentences_dev}/{sentences_test}; Annotation count {annotations_train}/{annotations_dev}/{annotations_test}')
        assert annotations_train + annotations_dev + annotations_test == final_datapoints

        info('Saving training data [1/3]')
        save_data_folders(train_out, machine_data[:eighth])
        info('Saving dev data [2/3]')
        save_pickle(dev_out, machine_data[eighth:ninth])
        info('Saving test data [3/3]')
        save_pickle(test_out, machine_data[ninth:])

        del machine_data

    info("Done")


if __name__ == "__main__":
    build_train_machine_data(test_mode)
