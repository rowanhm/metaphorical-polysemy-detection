from nltk.corpus import wordnet as wn
from src.shared.common import flatten


def filter_lemmas(lemmas):
    return [l for l in lemmas if l.name().lower() == l.name()]


def filter_synsets(synsets):
    filtered_synsets = []
    included_synsets = set()
    for synset in synsets:
        lemmas = synset.lemmas()
        if len(filter_lemmas(lemmas)) > 0 and synset.name() not in included_synsets:
            included_synsets.add(synset.name())
            filtered_synsets += [synset]
    return filtered_synsets


def get_synsets(word, pos_codes=None):

    if pos_codes is None:
        synsets_unfiltered = wn.synsets(word)
    else:
        synsets_unfiltered = flatten([wn.synsets(word, pos_wn_code) for pos_wn_code in pos_codes])

    filtered_synsets = filter_synsets(synsets_unfiltered)

    # Remove any synsets of acronyms
    synsets_with_acronyms_removed = []
    for synset in filtered_synsets:
        lemmas = [l.name() for l in synset.lemmas()]
        if not (word not in lemmas and word.upper() in lemmas):
            synsets_with_acronyms_removed += [synset]

    return synsets_with_acronyms_removed


def get_synset_names(word, pos_codes=None):
    return [synset.name() for synset in get_synsets(word, pos_codes)]
