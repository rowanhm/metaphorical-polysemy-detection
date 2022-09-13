# Process the annotation to a machine file
from src.shared.common import open_dict_csv, open_pickle, info, save_pickle
from src.shared.global_variables import mpd_verbs_csv_file, mpd_all_machine_file, mpd_verbs_machine_file, mpd_all_csv_file, \
    mpd_mml_csv_file, mpd_mml_machine_file, sense_ids_file, word_ids_file, mpd_oov_csv_file, \
    mpd_oov_machine_file


def build_machine_eval_data():
    info("Loading dictionaries")
    sense_dict = open_pickle(sense_ids_file)
    word_dict = open_pickle(word_ids_file)

    missing_senses = 0

    for (csv_file, machine_file) in [(mpd_verbs_csv_file, mpd_verbs_machine_file), (mpd_all_csv_file, mpd_all_machine_file),
                                     (mpd_mml_csv_file, mpd_mml_machine_file), (mpd_oov_csv_file, mpd_oov_machine_file)]:

        info(f'Processing {csv_file}')
        data = open_dict_csv(csv_file)
        data_bundled = dict()
        datapoints = 0

        for datapoint in data:
            word = datapoint['word']
            metaphor = datapoint['metaphor']
            sense = datapoint['sense']
            datapoint_id = datapoint['datapoint_id']

            if sense in sense_dict.keys():
                sense = sense_dict[sense]
            else:
                missing_senses += 1
                continue

            if metaphor == '':
                metaphor = None
            else:
                metaphor = int(metaphor)

            if word not in data_bundled.keys():
                data_bundled[word] = (word_dict[word], [], [], [])  # (word, ids, senses, metaphor_labels)

            # Add the sense
            (word_id, datapoint_ids, senses, metaphor_labels) = data_bundled[word]
            senses += [sense]
            metaphor_labels += [metaphor]
            datapoint_ids += [datapoint_id]
            data_bundled[word] = (word_id, datapoint_ids, senses, metaphor_labels)
            datapoints += 1

        # Flatten
        output = [d for d in data_bundled.values()]
        info(f'{datapoints} datapoints in {len(data_bundled.keys())} words; {missing_senses} datapoints excluded for missing sense')

        save_pickle(machine_file, output)


if __name__ == "__main__":
    build_machine_eval_data()
