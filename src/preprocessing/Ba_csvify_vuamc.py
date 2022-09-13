import xml.etree.ElementTree as et
import numpy as np

from src.shared.common import save_csv, info, save_pickle, open_pickle, save_text
from src.shared.global_variables import wn_pos, md_vuamc_vocab_file, md_vuamc_extracted_file, md_vuamc_raw_file, \
    sense_ids_file, md_vuamc_xml_file
from src.preprocessing.helper import get_synset_names


def extract_vuamc(exclude_conventional=False):

    vocab = set()
    sense_vocab = open_pickle(sense_ids_file).keys()

    root = et.parse(md_vuamc_raw_file).getroot()

    sentence_iter = root.iter('{http://www.tei-c.org/ns/1.0}s')

    sentences = []
    all_pos = []
    all_met = []
    all_words = []
    all_lemmas = []
    all_offsets = []
    sentence_ids = []
    all_wsd_indices = []

    data_distribution = dict({
        'verb': 0,
        'noun': 0,
        'adj': 0,
        'adv': 0,
        'excluded_pos': 0,
        'excluded_wn': 0,
        'excluded_lemma': 0,
        'excluded_no_word': 0,
        'excluded_sense': 0,
        'excluded_novel': 0,
        'excluded_no_score': 0
    })
    NOVELTY_THRESH = 0.2
    pos_include = dict({
        'AJ0': 'adj',
        'AJC': 'adj',
        'AJS': 'adj',
        'AV0': 'adv',
        'AVP': 'adv',
        'NN0': 'noun',
        'NN1': 'noun',
        'NN2': 'noun',
        # 'NP0': 'noun',
        'VBB': 'verb',
        'VBD': 'verb',
        'VBG': 'verb',
        'VBI': 'verb',
        'VBN': 'verb',
        'VBZ': 'verb',
        'VDB': 'verb',
        'VDD': 'verb',
        'VDG': 'verb',
        'VDI': 'verb',
        'VDN': 'verb',
        'VDZ': 'verb',
        'VHB': 'verb',
        'VHD': 'verb',
        'VHG': 'verb',
        'VHI': 'verb',
        'VHN': 'verb',
        'VHZ': 'verb',
        # 'VM0': 'verb',
        'VVB': 'verb',
        'VVD': 'verb',
        'VVG': 'verb',
        'VVI': 'verb',
        'VVN': 'verb',
        'VVZ': 'verb'
    })

    xml = ['<?xml version="1.0" encoding="UTF-8" ?>', '<corpus lang="en" source="vuamc">', '<text id="0">']
    wsd_uid_iter = 0

    for sentence_ind, sentence in enumerate(sentence_iter):

        sentence_ids += [sentence.attrib['n']]

        start_index = 0

        words = []
        lemmas = []
        metaphors = []
        psos = []
        offsets = []
        sentence_string = ""
        wsd_indices = []

        previous_space = False
        metaphor = False

        # Filter tokens
        skip_sentence = False
        sentence_flat = []
        for token_ind, token in enumerate(sentence):
            if token.tag == '{http://www.tei-c.org/ns/1.0}choice':
                #  Skip the 8 sentences with typos
                skip_sentence = True
                break
            elif token.tag in {'{http://www.tei-c.org/ns/1.0}hi', '{http://www.tei-c.org/ns/1.0}shift',
                               '{http://www.tei-c.org/ns/1.0}sic'}:
                # 'hi' is rendering details e.g italic, 'shift' is a change in register
                sentence_flat += [t for t in token]
            elif token.tag in {'{http://www.tei-c.org/ns/1.0}pb', '{http://www.tei-c.org/ns/1.0}ptr',
                               '{http://www.tei-c.org/ns/1.0}gap', '{http://www.tei-c.org/ns/1.0}pause',
                               '{http://www.tei-c.org/ns/1.0}vocal', '{http://www.tei-c.org/ns/1.0}incident'}:
                continue
            elif token.tag in {'{http://www.tei-c.org/ns/1.0}seg'}:
                # Skip 291 sentences with truncated word
                assert token.attrib['function'] == 'trunc'
                skip_sentence = True
                break
            else:
                sentence_flat += [token]

        if not skip_sentence:
            xml += [f'<sentence id="{sentence_ind}">']

            number_quotations = 0
            for token in sentence_flat:

                include_in_metaphor = False

                no_space = False
                if token.tag == '{http://www.tei-c.org/ns/1.0}c':
                    pos = "PUNCT"
                    if token.attrib['type'] == 'PUL':  # Open brackets
                        previous_space = True
                    else:
                        no_space = True
                else:
                    assert token.tag == '{http://www.tei-c.org/ns/1.0}w'

                    # Read basics
                    attributes = token.attrib
                    lemma = '_'.join(attributes['lemma'].split())
                    pos = attributes['type']

                # Determine metaphoricity
                metaphor_annotation = token.findall('{http://www.tei-c.org/ns/1.0}seg')

                if not metaphor_annotation:
                    metaphor_temp = 0
                    text = token.text
                else:
                    metaphor_temp = 1
                    text = token.text
                    if text is not None:
                        text = text.strip()
                    else:
                        text = ""

                    novelty_scores = []
                    for anno in metaphor_annotation:

                        # Don't include marker words e.g. 'like' as metaphors
                        if anno.attrib['function'] == 'mFlag':
                            metaphor_temp = 0
                        else:
                            assert anno.attrib['function'] == 'mrw'

                            if 'score' in anno.attrib.keys():
                                novelty_scores.append(float(anno.attrib['score']))

                        anno_text = anno.text
                        anno_tail = anno.tail
                        if anno_text is not None:
                            text += anno_text.strip()
                        if anno_tail is not None:
                            text += anno_tail.strip()

                    text += token.tail.strip()

                # Flip `" '
                if pos == 'PUNCT':
                    if token.attrib['type'] == "PUQ":
                        if text == '" ':
                            if number_quotations % 2 == 0:
                                # Even - this is opening
                                text = '"'
                                previous_space = True
                                number_quotations += 1
                    else:
                        assert token.attrib['type'] in {"PUN", "PUL", "PUR"}

                # Remove leading whitespace
                text = text.lstrip()

                if '\n' in text:
                    text = text.split('\n')[0]

                previous_space_temp = False
                if text[-1:] == ' ':
                    text = text.rstrip()
                    previous_space_temp = True

                # Add space or not
                if not no_space:
                    if text in {"'s", "'ve", "'d", "n't"} or pos == "POS":
                        no_space = True

                if metaphor and not no_space:  # Previous token was metaphor
                    previous_space = True  # Add a space

                if previous_space:
                    start_index += 1
                    sentence_string += ' '

                previous_space = previous_space_temp
                metaphor = metaphor_temp

                end_index = start_index + len(text)

                if pos == 'PUNCT':
                    lemma = text

                if pos in pos_include.keys():

                    coarse_pos = pos_include[pos]

                    word = '_'.join(text.split())

                    lemma_synsets = set(get_synset_names(lemma, wn_pos[coarse_pos]))
                    word_synsets = set(get_synset_names(word, wn_pos[coarse_pos]))
                    wsd_sense_options = lemma_synsets.union(word_synsets)

                    if len(wsd_sense_options) >= 1:  # Has a sense

                        sense_options_filtered = [sense for sense in wsd_sense_options if sense in sense_vocab]

                        if len(sense_options_filtered) >= 1:

                            if not exclude_conventional:

                                data_distribution[coarse_pos] += 1

                                # Add lemma to sentence so far
                                words += [word]
                                # if lemma not in all_wn_lemmas:
                                #     print()
                                lemmas += [lemma]
                                metaphors += [metaphor]
                                psos += [coarse_pos]
                                offsets += [(start_index, end_index)]
                                wsd_indices += [str(wsd_uid_iter)]
                                vocab.add(word)
                                include_in_metaphor = True

                            else:
                                novelty_found = True  # Defaults to max
                                if metaphor:
                                    if len(novelty_scores) == 0:
                                        novelty_found = False
                                    else:
                                        novelty = np.max(novelty_scores)  # Max to be conservative

                                if novelty_found:
                                    if (not metaphor) or (metaphor and novelty < NOVELTY_THRESH):
                                        data_distribution[coarse_pos] += 1

                                        # Add lemma to sentence so far
                                        words += [word]
                                        # if lemma not in all_wn_lemmas:
                                        #     print()
                                        lemmas += [lemma]
                                        metaphors += [metaphor]
                                        psos += [coarse_pos]
                                        offsets += [(start_index, end_index)]
                                        wsd_indices += [str(wsd_uid_iter)]
                                        vocab.add(word)
                                        include_in_metaphor = True
                                    else:
                                         data_distribution["excluded_novel"] += 1
                                else:
                                    data_distribution["excluded_no_score"] += 1
                        else:
                            data_distribution["excluded_sense"] += 1

                    else:
                        data_distribution["excluded_wn"] += 1

                else:
                    data_distribution["excluded_pos"] += 1

                # Update for next iteration
                sentence_string += text
                assert sentence_string[start_index:end_index] == text
                start_index = end_index

                # Add for WSD component
                text = text.replace('"', '``')
                text = text.replace('&', '&amp;')
                text = text.replace("'", "&apos;")
                lemma = lemma.replace('"', '``')
                lemma = lemma.replace('&', '&amp;')
                lemma = lemma.replace("'", "&apos;")
                if include_in_metaphor:
                    xml += [f'<instance id="{wsd_uid_iter}" lemma="{lemma}">{text}</instance>']
                    wsd_uid_iter += 1
                else:
                    xml += [f'<wf lemma="{lemma}">{text}</wf>']

            assert len(psos) == len(metaphors)
            assert len(psos) == len(lemmas)
            assert len(psos) == len(words)
            assert len(psos) == len(offsets)
            assert len(psos) == len(wsd_indices)

            if len(offsets) > 0:
                sentences += [sentence_string]
                all_pos += [psos]
                all_met += [metaphors]
                all_lemmas += [lemmas]
                all_words += [words]
                all_offsets += [offsets]
                all_wsd_indices += [wsd_indices]

            xml += ['</sentence>']

    info(f"{data_distribution['excluded_pos']} excluded for wrong POS; {data_distribution['excluded_wn']} for no synsets; {data_distribution['excluded_lemma']} for divergent word/lemma synsets; {data_distribution['excluded_no_word']} for no lemma corresponding to word; {data_distribution['excluded_sense']} for sense out of vocab; {data_distribution['excluded_no_score']} for no novelty score; {data_distribution['excluded_novel']} for novelty")

    # Format into dataframe
    data = []
    metaphor_count = 0
    literal_count = 0
    for sent_id, (_, sentence, offsets, words, lemmas, metaphors, psos, wsd_indices) in \
            enumerate(zip(sentence_ids, sentences, all_offsets, all_words, all_lemmas, all_met, all_pos, all_wsd_indices)):
        for sentence_ind, ((start, end), lemma, word, met, pos, wsd_index) in enumerate(zip(offsets, lemmas, words, metaphors, psos, wsd_indices)):
            data += [dict({
                'datapoint_id': f'vuamc.{sent_id}.{sentence_ind}',
                'start_offset': start,
                'end_offset': end,
                'pos': pos,
                'lemma': lemma,
                'word': word,
                'metaphor': met,
                'sentence': sentence,
                'wsd_index': wsd_index
            })]
            if met:
                metaphor_count += 1
            else:
                literal_count += 1

            sentence = ''  # Only store the sentence for the first datapoint from each sentence

    info(f'{metaphor_count} metaphors and {literal_count} literals')
    info(f'{len(data)} datapoints in {len(sentences)} sentences.')
    save_csv(md_vuamc_extracted_file, data)

    info(f'{len(vocab)} items in vuamc vocab')
    save_pickle(md_vuamc_vocab_file, vocab)

    xml += ['</text>', '</corpus>']
    save_text(md_vuamc_xml_file, xml)
    info('Done')

if __name__ == "__main__":
    extract_vuamc()
