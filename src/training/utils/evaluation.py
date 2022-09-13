import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_recall_curve, roc_auc_score, \
    average_precision_score, label_ranking_loss, label_ranking_average_precision_score, ndcg_score

from src.shared.common import flatten


def threshold_predict(labels, probabilities):
    precs, recs, thresholds = precision_recall_curve(labels, probabilities)
    f1s = [2 * p * r / (r + p) if p + r > 0 else 0 for p, r in zip(precs, recs)]
    best_threshold = thresholds[np.argmax(f1s)]
    predictions = [prob >= best_threshold for prob in probabilities]
    return predictions, best_threshold


def filter_unlabelled(all_datapoints, all_probabilities, all_labels):
    # Remove unlabelled items
    all_labels_filtered = []
    all_datapoints_filtered = []
    all_probabilities_filtered = []
    for labels, probabilities, datapoints in zip(all_labels, all_probabilities, all_datapoints):
        labels_filtered = []
        datapoints_filtered = []
        probabilities_filtered = []
        for label, probability, datapoint in zip(labels, probabilities, datapoints):
            if label is not None:
                labels_filtered += [label]
                probabilities_filtered += [probability]
                datapoints_filtered += [datapoint]
        assert len(labels_filtered) == len(datapoints_filtered)
        assert len(labels_filtered) == len(probabilities_filtered)
        if len(labels_filtered) > 0:
            all_labels_filtered += [labels_filtered]
            all_datapoints_filtered += [datapoints_filtered]
            all_probabilities_filtered += [probabilities_filtered]
    return all_datapoints_filtered, all_probabilities_filtered, all_labels_filtered

def evaluate(all_datapoints, all_probabilities, all_labels):
    all_datapoints_filtered, all_probabilities_filtered, all_labels_filtered = filter_unlabelled(all_datapoints, all_probabilities, all_labels)

    all_probabilities_flat = flatten(all_probabilities_filtered)
    all_datapoints_flat = flatten(all_datapoints_filtered)
    all_labels_flat = flatten(all_labels_filtered)
    assert len(all_probabilities_flat) == len(all_datapoints_flat)
    assert len(all_probabilities_flat) == len(all_labels_flat)

    result_dict = dict()
    if len(all_datapoints_flat) > 0:
        # First, calculate general stats with no thresholding
        predictions_raw = [p >= 0.5 for p in all_probabilities_flat]
        prec_raw, rec_raw, f1_raw, _ = precision_recall_fscore_support(all_labels_flat, predictions_raw,
                                                                       average='binary')
        acc_raw = accuracy_score(all_labels_flat, predictions_raw)

        # Next, calculate by maxing a single sliding boundary
        predictions_single, threshold = threshold_predict(all_labels_flat, all_probabilities_flat)
        prec_single, rec_single, f1_single, _ = precision_recall_fscore_support(all_labels_flat,
                                                                                predictions_single,
                                                                                average='binary')
        acc_single = accuracy_score(all_labels_flat, predictions_single)

        # Third, calculate for per-word thresholding
        predictions_multi = []
        for (labels, probabilities) in zip(all_labels_filtered, all_probabilities_filtered):
            if all([l == 1 for l in labels]) or all([l == 0 for l in labels]):
                predictions_multi += labels  # In this case the threshold just maxes out
            else:
                predictions, _ = threshold_predict(labels, probabilities)
                predictions_multi += predictions
        prec_multi, rec_multi, f1_multi, _ = precision_recall_fscore_support(all_labels_flat,
                                                                             predictions_multi,
                                                                             average='binary')
        acc_multi = accuracy_score(all_labels_flat, predictions_multi)

        # Fourth, calc metrics which don't care about threshold
        overall_roc_auc = roc_auc_score(all_labels_flat, all_probabilities_flat)
        overall_pr_auc = average_precision_score(all_labels_flat, all_probabilities_flat)

        # Finally, calculate ranking metrics
        word_pr_aucs = []
        word_roc_aucs = []
        ranking_losses = []
        ranking_precisions = []
        ndcgs = []
        for (labels, probabilities) in zip(all_labels_filtered, all_probabilities_filtered):
            #precision, recall, _ = precision_recall_curve(labels, probabilities)
            #pr_auc = auc(recall, precision)
            #pr_aucs += [pr_auc]
            if not (all([l == 1 for l in labels]) or all([l == 0 for l in labels])):
                #word_pr_aucs += [1.0]
                #word_roc_aucs += [1.0]
                #ranking_losses += [0.0]
                #ranking_precisions += [1.0]
                #ndcgs += [1.0]
            #else:
                word_pr_aucs += [average_precision_score(labels, probabilities)]
                word_roc_aucs += [roc_auc_score(labels, probabilities)]
                ranking_losses += [label_ranking_loss([labels], [probabilities])]
                ranking_precisions += [label_ranking_average_precision_score([labels], [probabilities])]
                ndcgs += [ndcg_score([labels], [probabilities])]
        ndcg = np.mean(ndcgs)
        ranking_loss = np.mean(ranking_losses)
        ranking_precision = np.mean(ranking_precisions)
        word_roc_auc = np.mean(word_roc_aucs)
        word_pr_auc = np.mean(word_pr_aucs)

        result_dict['ndcg'] = ndcg
        result_dict['ranking_loss'] = ranking_loss
        result_dict['ranking_precision'] = ranking_precision
        result_dict['word_roc_auc'] = word_roc_auc
        result_dict['word_pr_auc'] = word_pr_auc

        result_dict['overall_roc_auc'] = overall_roc_auc
        result_dict['overall_pr_auc'] = overall_pr_auc

        result_dict['f1.raw'] = f1_raw
        result_dict['f1.single_thresh'] = f1_single
        result_dict['f1.multi_thresh'] = f1_multi

        result_dict['accuracy.raw'] = acc_raw
        result_dict['accuracy.single_thresh'] = acc_single
        result_dict['accuracy.multi_thresh'] = acc_multi

        result_dict['recall.raw'] = rec_raw
        result_dict['recall.single_thresh'] = rec_single
        result_dict['recall.multi_thresh'] = rec_multi

        result_dict['precision.raw'] = prec_raw
        result_dict['precision.single_thresh'] = prec_single
        result_dict['precision.multi_thresh'] = prec_multi

        full_output = [('datapoint_id', 'p_met')] + list(zip(all_datapoints_flat, all_probabilities_flat))

    else:
        # Stub; dataset not annotated
        result_dict['ndcg'] = None
        result_dict['ranking_loss'] = None
        result_dict['ranking_precision'] = None
        result_dict['word_roc_auc'] = None
        result_dict['word_pr_auc'] = None
        result_dict['overall_roc_auc'] = None
        result_dict['overall_pr_auc'] = None
        result_dict['f1.raw'] = None
        result_dict['f1.single_thresh'] = None
        result_dict['f1.multi_thresh'] = None
        result_dict['accuracy.raw'] = None
        result_dict['accuracy.single_thresh'] = None
        result_dict['accuracy.multi_thresh'] = None
        result_dict['recall.raw'] = None
        result_dict['recall.single_thresh'] = None
        result_dict['recall.multi_thresh'] = None
        result_dict['precision.raw'] = None
        result_dict['precision.single_thresh'] = None
        result_dict['precision.multi_thresh'] = None
        full_output = []

    return result_dict, full_output
