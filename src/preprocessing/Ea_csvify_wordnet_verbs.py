from nltk.corpus import wordnet as wn

from src.shared.common import open_dict_csv, info, save_csv
from src.shared.global_variables import mpd_verbs_raw_file, mpd_verbs_csv_file, wn_pos_reverse
from src.preprocessing.helper import filter_synsets, get_synsets


def build_verb_test_data():
    data = open_dict_csv(mpd_verbs_raw_file, delimiter='\t')
    readable_data = []

    datapoint_count = 0
    word_to_sense_options = dict()

    excluded_proper_nouns = 0

    for datapoint in data[:-2]:  # Exclude last two rows of metaphor verb annotation

        word = datapoint['term']
        sense_label = datapoint['sense']

        assert type(sense_label) is str

        synset = wn.synset('.'.join(sense_label.split('#')))
        if len(filter_synsets([synset])) == 0:
            excluded_proper_nouns += 1
            continue

        sense = synset.name()
        pos = wn_pos_reverse[synset.pos()]
        assert pos == 'verb'

        metaphor_label = datapoint['class']
        if metaphor_label == 'metaphorical':
            metaphor = 1
        else:
            assert metaphor_label == 'literal'
            metaphor = 0

        word = ' '.join(word.split('_'))

        if word in word_to_sense_options.keys():
            word_to_sense_options[word] += [sense]
        else:
            word_to_sense_options[word] = [sense]

        readable_data += [dict({
            'datapoint_id': f'verbs.{datapoint_count}',
            'word': word,
            'sense': sense,
            'metaphor': metaphor,
            'pos': pos,
            'definition': f'[{"; ".join([l.name() for l in synset.lemmas()])}] {synset.definition()}'
        })]
        datapoint_count += 1

    info(f'{len(readable_data)} word-sense pairs, in {len(word_to_sense_options.keys())} words. {excluded_proper_nouns} excluded for capitalisation')

    # NB, verbs with no example usages were excluded from this annotation
    extra_count = 0
    for word, labeled_senses in word_to_sense_options.items():
        word_synsets = get_synsets(word)
        for synset in word_synsets:
            sense = synset.name()
            pos = wn_pos_reverse[synset.pos()]
            if sense not in labeled_senses:
                readable_data += [dict({
                    'datapoint_id': f'verbs.{datapoint_count}',
                    'word': word,
                    'sense': sense,
                    'metaphor': '',
                    'pos': pos,
                    'definition': f'[{"; ".join([l.name() for l in synset.lemmas()])}] {synset.definition()}'
                })]
                datapoint_count += 1
                extra_count += 1

        # Make sure the labelled senses are sorted out right
        word_synset_names = set([synset.name() for synset in word_synsets])
        for sense_name in labeled_senses:
            assert sense_name in word_synset_names

    info(f'Adding {extra_count} senses not included in annotation')

    save_csv(mpd_verbs_csv_file, readable_data)


if __name__ == "__main__":
    build_verb_test_data()
