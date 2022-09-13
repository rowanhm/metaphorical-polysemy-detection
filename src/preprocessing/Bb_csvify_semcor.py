import nltk
from nltk.corpus import semcor as sc
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import WordNetError
from sacremoses import MosesDetokenizer

from src.shared.common import save_csv, info, warn, save_pickle, open_pickle
from src.shared.global_variables import wn_pos, wsd_semcor_extracted_file, wsd_semcor_vocab_file, sense_ids_file
from src.preprocessing.helper import get_synset_names


def extract_semcor():

    # all_wn_lemmas = [k for k in open_pickle_bz2(os.path.join(machine_data_dir, 'lemmas_2_ids.pkl.bz2')).keys()]
    vocab = set()
    sense_vocab = open_pickle(sense_ids_file).keys()

    detokenizer = MosesDetokenizer(lang='en')
    testing = False

    skip_ambiguous = True

    sentences = []

    all_offsets = []
    all_lemmas = []
    all_senses = []
    all_pos = []
    all_words = []

    data_distribution = dict({
        'verb': 0,
        'noun': 0,
        'adj': 0,
        'adv': 0,
        'excluded': 0,
        'excluded_options': 0,
        'excluded_sense': 0
    })
    pos_include = dict({
        'RB': 'adv',
        'VBN': 'verb',
        'JJ': 'adj',
        #'NE': '',
        'NN': 'noun',
        'VB': 'verb',
        'NNS': 'noun',
        'VBD': 'verb',
        'VBG': 'verb'
    })

    info("Extracting sentences and senses")

    skipped_semicolon = []
    skipped_wordnet = []
    skipped_lemma = []
    skipped_ambiguous = []

    # Go though all the data
    for (tokens, synsets) in zip(sc.sents(), sc.tagged_sents(tag='both')):  # NB: This is semcor-all and semcor-verbs combined

        if testing and len(sentences) > 10:
            break

        # Preprocess
        tokens = ' '.join(tokens).replace('``', '\"').replace("''", '\"').replace('`', "\'").split()
        sentence = detokenizer.detokenize(tokens)

        offsets = []
        senses = []
        psos = []
        wordses = []

        index_start = 0
        index_end = 0
        sense = None

        # Go through senses in each sentence
        for i, t in enumerate(synsets):

            labelled = False
            if type(t[0]) == nltk.tree.Tree:
                if type(t.label()) == str:
                    if t.label() != 'NE':

                        labelled = True
                else:
                    labelled = True

            words = t.leaves()

            if labelled:
                assert type(words) == list
                sense = t.label()
                pos = t[0].label()

                if type(sense) == str:

                    if not skip_ambiguous:
                        sense_split = sense.split(".")
                        assert len(sense_split) == 3

                        if ";" in sense_split[-1]:
                            labelled = False
                            skipped_semicolon += [sense]
                        else:
                            try:
                                lemmas_for_sense = wn.synset(sense).lemmas()
                                matching_lemmas = []
                                this_lemma = sense_split[0].lower() #.replace('.', '')
                                # Get the lemma for the sense
                                for lemma in lemmas_for_sense:
                                    if this_lemma == lemma.name().lower():
                                        matching_lemmas += [lemma]

                                if len(matching_lemmas) != 1:
                                    info(f'{words}; {pos}; {sense}; {lemmas_for_sense} ({sentence})')
                                    labelled = False
                                    skipped_lemma += [sense]
                                else:
                                    sense = matching_lemmas[0]

                            except WordNetError:
                                labelled = False
                                skipped_wordnet += [sense]

                    else:
                        labelled = False
                        skipped_ambiguous += [sense]

            # Find end index and check alignment
            for k, word in enumerate(words):

                word = word.replace('``', '\"').replace("''", '\"').replace('`', "\'")
                index_temp = index_end + len(word)

                assert sentence[index_end:index_temp] == word

                if k < len(words)-1:
                    if sentence[index_temp:index_temp + 1] == ' ':
                        index_temp += 1

                index_end = index_temp

            if labelled:
                if pos in pos_include.keys():
                    coarse_pos = pos_include[pos]

                    lemma = sense.name()
                    word = "_".join(word.split())

                    # assert lemma in all_wn_lemmas
                    lemma_synsets = set(get_synset_names(lemma, wn_pos[coarse_pos]))
                    word_synsets = set(get_synset_names(lemma, wn_pos[coarse_pos]))
                    sense_options_wsd = lemma_synsets.union(word_synsets)

                    if len(sense_options_wsd) > 1 and sense.synset().name() in sense_options_wsd:

                        sense_options_filtered = [sense for sense in sense_options_wsd if sense in sense_vocab]

                        if len(sense_options_filtered) >= 1:
                            data_distribution[coarse_pos] += 1

                            offsets += [(index_start, index_end)]
                            senses += [sense]
                            psos += [coarse_pos]
                            wordses += [word]

                            vocab.add(word)
                        else:
                            data_distribution["excluded_sense"] += 1
                    else:
                        data_distribution["excluded_options"] += 1
                else:
                    data_distribution["excluded"] += 1

            if sentence[index_end:index_end + 1] == ' ':
                index_end += 1

            index_start = index_end

        assert len(offsets) == len(senses)
        assert len(offsets) == len(psos)
        assert len(offsets) == len(senses)
        assert len(offsets) == len(wordses)
        if len(offsets) > 0:
            sentences += [sentence]
            all_offsets += [offsets]
            all_lemmas += [[sense.name() for sense in senses]]
            all_senses += [[sense.synset().name() for sense in senses]]
            all_pos += [psos]
            all_words += [wordses]

    if not skip_ambiguous:
        warn(f"Skipped {len(skipped_semicolon)} words for multiple annotation, {len(skipped_wordnet)} for invalid WN 3.0 annotation, and {len(skipped_lemma)} for ambiguous lemma.")
    else:
        warn(f"Skipped {len(skipped_ambiguous)} ambiguous annotations.")

    info(f"{data_distribution['excluded']} excluded for wrong POS; {data_distribution['excluded_options']} for singleton option; {data_distribution['excluded_sense']} for sense out of vocab")

    # Format into dataframe
    data = []
    for sent_id, (sentence, offsets, lemmas, words, senses, psos) in enumerate(zip(sentences, all_offsets, all_lemmas,
                                                                                   all_words, all_senses, all_pos)):
        for i, ((start, end), lemma, word, sense, pos) in enumerate(zip(offsets, lemmas, words, senses, psos)):
            data += [dict({
                'datapoint_id': f'semcor.{sent_id}.{i}',
                'start_offset': start,
                'end_offset': end,
                'pos': pos,
                'lemma': lemma,
                'word': word,
                'sense': sense,
                'sentence': sentence
            })]
            sentence = ''  # Only store the sentence for the first datapoint from each sentence

    info(f'{len(data)} datapoints in {len(sentences)} sentences.')
    save_csv(wsd_semcor_extracted_file, data)

    info(f'{len(vocab)} items in semcor vocab')
    save_pickle(wsd_semcor_vocab_file, vocab)


if __name__ == "__main__":
    extract_semcor()
