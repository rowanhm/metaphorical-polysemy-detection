import torch
from nltk.corpus import wordnet as wn

from src.shared.common import open_pickle, save_pickle, info
from src.shared.global_variables import sense_ids_file, sense_adjacency_file
from src.preprocessing.helper import filter_synsets


def build_adjacency_matrix():

    info("Loading & formatting data")
    senses_to_ids = open_pickle(sense_ids_file)
    ids_to_senses = dict()
    for sense, sense_id in senses_to_ids.items():
        assert sense_id not in ids_to_senses.keys()
        ids_to_senses[sense_id] = sense

    num_senses = len(ids_to_senses) + 1  # +1 for null 0th item
    ascending_ids = sorted([k for k in ids_to_senses.keys()])
    assert ascending_ids == list(range(1, num_senses))

    info(f"Computing adjacencies for {num_senses} senses")
    current_id = 1
    adjacency_coords = []
    values = []

    for sense_id in ascending_ids:

        assert sense_id == current_id
        current_id += 1

        num_hypernyms = 0

        sense = ids_to_senses[sense_id]
        synset = wn.synset(sense)
        assert synset in filter_synsets([synset])
        for hypernym in filter_synsets(synset.hypernyms()):
            hypernym_name = hypernym.name()
            hypernym_id = senses_to_ids[hypernym_name]

            adjacency_coords += [[sense_id, hypernym_id]]
            num_hypernyms += 1

        if num_hypernyms > 0:
            values += [1/num_hypernyms] * num_hypernyms

    assert len(values) == len(adjacency_coords)

    info("Building sparse tensor")
    sense_adjacencies_tensor = torch.sparse_coo_tensor(torch.tensor(adjacency_coords).t(),
                                                       values,
                                                       (num_senses, num_senses)).float().cpu()

    info("Saving")
    save_pickle(sense_adjacency_file, sense_adjacencies_tensor)


if __name__ == "__main__":
    build_adjacency_matrix()
