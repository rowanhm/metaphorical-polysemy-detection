from nltk.corpus import wordnet as wn

from src.shared.common import info, save_csv
from src.shared.global_variables import wn_pos_reverse, mpd_all_csv_file
from src.preprocessing.helper import filter_synsets, filter_lemmas


def build_final_eval_data():

    readable_data = []
    singleton_lemmas = 0
    datapoint_count = 0

    for synset in filter_synsets(wn.all_synsets()):
        lemmas = filter_lemmas(synset.lemmas())
        pos = wn_pos_reverse[synset.pos()]
        sense = synset.name()

        for lemma in lemmas:
            # If the lemmas has multiple associated synsets
            if len(filter_synsets(wn.synsets(lemma.name()))) > 1:

                # word = ' '.join(lemma.name().split('_'))
                word = lemma.name()

                readable_data += [dict({
                    'datapoint_id': f'all.{datapoint_count}',
                    'word': word,
                    'sense': sense,
                    'metaphor': '',
                    'pos': pos,
                    'definition': f'[{"; ".join([l.name() for l in synset.lemmas()])}] {synset.definition()}'
                })]
                datapoint_count += 1

            else:
                singleton_lemmas += 1

    assert datapoint_count == len(readable_data)
    info(f'{datapoint_count} word-sense pairs ({singleton_lemmas} excluded singleton lemma entries)')
    save_csv(mpd_all_csv_file, readable_data)


if __name__ == "__main__":
    build_final_eval_data()
