import os
import torch
from gensim.models import KeyedVectors
from nltk.corpus import wordnet as wn
import numpy as np
from sklearn.decomposition import TruncatedSVD

from src.shared.common import info, save_pickle
from src.shared.global_variables import raw_data_dir, extracted_data_dir, sense_embs_file, sense_ids_file, seed

torch.manual_seed(seed)
from src.preprocessing.helper import filter_lemmas


def build_sense_tensor():
    ares_file = os.path.join(extracted_data_dir, 'ares_embeddings.bin')
    if os.path.isfile(ares_file):
        info("Opening ARES sense embeddings from binary")
        ares_sense_embs = KeyedVectors.load_word2vec_format(ares_file, binary=True)
    else:
        info("Opening ARES sense embeddings from text")
        ares_sense_embs = KeyedVectors.load_word2vec_format(
            os.path.join(raw_data_dir, 'ares_embedding/ares_bert_large.txt'), binary=False)
        ares_sense_embs.save_word2vec_format(ares_file, binary=True)

    info("Building sense dictionary and extracting embeddings")
    sense_dict = dict()
    ares_original = []

    index = 1

    included_synsets = 0
    excluded_synsets = 0
    for synset in wn.all_synsets():

        ares_embeddings = []

        for lemma in filter_lemmas(synset.lemmas()):
            lemma_code = lemma.key()
            if ares_sense_embs.has_index_for(lemma_code):
                ares_embeddings += [ares_sense_embs[lemma_code]]

        if len(ares_embeddings) > 0:  # and len(image_embeddings) > 0:  # ares_synset_embedding is not None:
            synset_name = synset.name()
            sense_dict[synset_name] = index

            ares_original += [np.mean(ares_embeddings, axis=0)]
            index += 1

            included_synsets += 1
        else:
            excluded_synsets += 1

    assert len(ares_original) == included_synsets
    info(f'{included_synsets} synsets included and {excluded_synsets} excluded')

    info("Reducing dimentionality of ARES")
    dimentionality = 768
    svd = TruncatedSVD(n_components=dimentionality)
    embeddings = svd.fit_transform(ares_original)

    # Add the mask element
    embeddings = np.insert(embeddings, 0, np.zeros([dimentionality], dtype='single'), axis=0)

    info("Converting to tensor and saving")
    sense_embeddings_tensor = torch.tensor(embeddings).cpu()

    save_pickle(sense_ids_file, sense_dict)
    save_pickle(sense_embs_file, sense_embeddings_tensor)


if __name__ == "__main__":
    build_sense_tensor()
