import pandas as pd

from src.training.utils.file_editing import FileEditor
from src.shared.common import info
from src.postprocessing.significance import significance


def extract_results(experiment_id):

    info('Loading Results')
    file_editor = FileEditor(experiment_id)
    results = file_editor.get_results()
    params = file_editor.get_params()

    results['mean.f1.dev'] = ((results['semcor.f1.dev'] + results['vuamc.f1.dev']) / 2)
    # Manually add dev readings for the MPD baselines
    results.at['random', 'mean.f1.dev'] = 1.0
    results.at['majority', 'mean.f1.dev'] = 1.0
    results.at['melbert', 'mean.f1.dev'] = 1.0

    results['mean.f1.test'] = ((results['semcor.f1.test'] + results['vuamc.f1.test']) / 2)
    # Manually add dev readings for the MPD baselines
    results.at['random', 'mean.f1.test'] = 1.0
    results.at['majority', 'mean.f1.test'] = 1.0
    results.at['melbert', 'mean.f1.test'] = 1.0

    info('Joining results with the parameters')
    results = pd.merge(params, results, how="right", on="param_id")

    info('Adding the option columns')
    results['WSD_options'] = [None] * len(results.index)
    results['MPD_options'] = [None] * len(results.index)
    results['MD_options'] = [None] * len(results.index)
    for index, row in results.iterrows():

        # WSD
        model = row['MODEL']
        if model == 'serial.sota.baseline':
            if row['ALPHA'] == 0:
                results.at[index, 'WSD_options'] = 'EWISER (WSD)'
            else:
                results.at[index, 'WSD_options'] = 'EWISER (WSD + MD)'
        elif model == 'serial.baseline.baseline':
            if row['ALPHA'] == 0:
                results.at[index, 'WSD_options'] = 'Baseline (WSD)'
            else:
                results.at[index, 'WSD_options'] = 'Baseline (WSD + MD)'

        # MD
        if type(model) is str:
            if model.split('.')[0] == 'serial':
                results.at[index, 'MD_options'] = 'LVM'
            elif model == 'met.sota':
                results.at[index, 'MD_options'] = 'MelBERT'
            elif model == 'met.baseline':
                results.at[index, 'MD_options'] = 'Baseline'

        # MPD
        if model == 'serial.sota.baseline':
            results.at[index, 'MPD_options'] = 'Ours (with EWISER WSD)'
        elif model == 'serial.baseline.baseline':
            results.at[index, 'MPD_options'] = 'Ours (with Baseline WSD)'
        elif index == 'random':
            results.at[index, 'MPD_options'] = 'Random'
        elif index == 'majority':
            results.at[index, 'MPD_options'] = 'Majority'
        elif index == 'melbert':
            results.at[index, 'MPD_options'] = 'MelBERT'

    info('Extracting best results')
    md_results = retrieve_results(results, 'MD_options', 'vuamc.f1.dev', ['vuamc.f1.test'])
    wsd_results = retrieve_results(results, 'WSD_options', 'semcor.f1.dev', ['semcor.f1.test'])
    mpd_results = retrieve_results(results, 'MPD_options', 'mean.f1.dev', ['verbs.f1.raw', 'verbs.word_roc_auc',
                                                                           'selection.f1.raw', 'selection.word_roc_auc',
                                                                           'random.f1.raw', 'random.word_roc_auc'])

    info('Printing latex tables')
    print_latex_table(md_results, sig_figs=3)
    print_latex_table(wsd_results, sig_figs=3)
    print_latex_table(mpd_results, sig_figs=2)

    info('Computing significance of MPD experiments')
    # Iterate through comparing best of ours to best of baselines
    for column in mpd_results.columns.values[1:]:
        if mpd_results.loc['Ours (with EWISER WSD)', column] > mpd_results.loc['Ours (with Baseline WSD)', column]:
            ours = 'Ours (with EWISER WSD)'
        else:
            ours = 'Ours (with Baseline WSD)'
        if mpd_results.loc['Random', column] > mpd_results.loc['Majority', column]:
            baseline = 'Random'
        else:
            baseline = 'Majority'
        if mpd_results.loc['MelBERT', column] > mpd_results.loc[baseline, column]:
            baseline = 'MelBERT'
        metric_code = 'mpd.' + '.'.join(column.split('.')[:2])
        v1_pred = mpd_results.loc[ours, column]
        v2_pred = mpd_results.loc[baseline, column]
        expected_difference = v1_pred - v2_pred
        if expected_difference > 0:
            print_significance(experiment_id, mpd_results, ours, baseline, metric=metric_code,
                               expected_difference=expected_difference)
        else:
            # Baseline outperforms ours
            expected_difference *= -1
            print_significance(experiment_id, mpd_results, baseline, ours, metric=metric_code,
                               expected_difference=-expected_difference)

    info('Computing significance of MD experiments')
    print_significance(experiment_id, md_results, 'LVM', 'MelBERT', metric='md')
    print_significance(experiment_id, md_results, 'MelBERT', 'Baseline', metric='md')
    print_significance(experiment_id, md_results, 'LVM', 'Baseline', metric='md')

    info('Computing significance of WSD experiments')
    print_significance(experiment_id, wsd_results, 'EWISER (WSD)', 'EWISER (WSD + MD)', metric='wsd')
    print_significance(experiment_id, wsd_results, 'Baseline (WSD)', 'Baseline (WSD + MD)', metric='wsd')


def print_significance(experiment_id, results, model_1, model_2, metric, expected_difference=None):
    # Get param ids
    model_1_param_id = results.loc[model_1, 'param_id']
    model_2_param_id = results.loc[model_2, 'param_id']
    sig = significance(experiment_id, model_1_param_id, model_2_param_id, metric=metric, expected_difference=None)
    print(f'{model_1} is higher than {model_2} under metric {metric}, p={sig}')


def print_latex_table(results, sig_figs=3):
    output_string = f"""\\begin{{table}}
    \\centering
    \\begin{{tabular}}{{l{'c'*(len(results.columns)-1)}}} \\toprule
        {' & '.join(results.columns.values)} \\\\ \\midrule"""
    for index, row in results.iterrows():
        print(row[0])
        row_items = row.values
        row_items = [index] + [stringify(val, sig_figs) for val in row_items[1:]]  # exclude row[0] as this is modelname
        output_string += f'\n        {" & ".join(row_items)} \\\\'
    output_string += """ \\bottomrule
    \end{tabular}
    \caption{}
    \label{tab:}
\end{table}"""
    print(output_string)


def stringify(val, sigfigs):
    rounded = round(val, sigfigs)
    formatted = '{:.{}f}'.format(rounded, sigfigs)
    return '$' + formatted[1:] + '$'


def retrieve_results(results, options_column, selection_column, result_columns):
    options = [opt for opt in results[options_column].unique() if opt is not None]
    columns = ['param_id', selection_column]+result_columns
    empty_table = [[''] + [0.0] * (len(columns)-1)] * len(options)
    retrieved_results = pd.DataFrame(data=empty_table, index=options, columns=columns)
    for index, row in results.iterrows():
        model_type = row[options_column]
        if model_type in options:
            dev_result = row[selection_column]
            if dev_result > retrieved_results.loc[model_type,selection_column]:
                # Update
                retrieved_results.at[model_type, 'param_id'] = index
                retrieved_results.at[model_type, selection_column] = dev_result
                for result_column in result_columns:
                    retrieved_results.at[model_type, result_column] = row[result_column]

    # Reshape the output
    retrieved_results = retrieved_results.drop(selection_column, axis=1)
    return retrieved_results


if __name__ == "__main__":
    extract_results(11)
