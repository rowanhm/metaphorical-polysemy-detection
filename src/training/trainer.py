import math
import time
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd

from src.models.components.wsd_models.wsd_precomputed import WSD_Precomputed
from src.training.utils.evaluation import evaluate
from src.shared.common import info
from src.shared.global_variables import seed
from src.training.data_loaders.joint_loader import JointLoader
from src.training.utils.trainer_utils import improved
from src.models.wrappers.complete_serial import SerialModel
from src.models.base import BaseMD, BaseWSD, BaseMPD, BaseComplete

torch.manual_seed(seed)


class Trainer:

    def __init__(self, train_loaders, dev_loaders, test_loaders, eval_loaders, runtime=None):

        self.train_loaders = train_loaders
        self.dev_loaders = dev_loaders
        self.test_loaders = test_loaders
        self.eval_loaders = eval_loaders

        self.joint_train_loader = JointLoader(loader_dict=self.train_loaders)

        # Timer
        if runtime is not None:
            self.deadline = runtime * 60 * 60 + time.time()
            self.time_remaining = self.positive_training_time
        else:
            self.time_remaining = lambda *_: True

        # Loss criterion
        met_criterion_raw = torch.nn.BCELoss(reduction='none')
        self.met_criterion = lambda log_probs, labels: met_criterion_raw(torch.exp(log_probs), labels)
        self.sense_criterion = torch.nn.NLLLoss(reduction='none')

    def train_and_eval(self, model, learning_rate, learning_rate_divisor, alpha, stopping_iterations, eval_steps, test=False, model_name=''):
        trained = self.train(model, learning_rate=learning_rate, learning_rate_divisor=learning_rate_divisor,
                             alpha=alpha, stopping_iterations=stopping_iterations, eval_steps=eval_steps, test=test,
                             model_name=model_name)

        if trained:
            info('Computing results')
            results_dict = dict()
            # First compute eval results
            eval_results_frame, eval_output = self.predict(model, self.eval_loaders)
            info(f'\n{eval_results_frame}')
            for dataset in eval_results_frame.index:
                for data_type in eval_results_frame.columns:
                    results_dict[f'{dataset}.{data_type}'] = eval_results_frame[data_type][dataset]

            # Now compute train/test/dev results
            outputs = dict()
            for (subset, loaders) in [('train', self.train_loaders), ('dev', self.dev_loaders),
                                      ('test', self.test_loaders)]:
                if not test:
                    results_frame, output_dict = self.overall_results(model, loaders)
                    results_frame.columns.name = subset.upper()
                    info(f'\n{results_frame}')
                else:
                    # Stub data
                    null_result = dict({
                        'loss': None,
                        'f1': None,
                        'accuracy': None,
                        'recall': None,
                        'precision': None
                    })
                    results = dict({
                        'vuamc': null_result,
                        'semcor': null_result
                    })
                    output_dict = dict({
                        'vuamc': ([], []),
                        'semcor': ([], [])
                    })
                    results_frame = pd.DataFrame.from_dict(results, orient='index')
                outputs[subset] = output_dict

                for dataset in results_frame.index:
                    for data_type in results_frame.columns:
                        results_dict[f'{dataset}.{data_type}.{subset}'] = results_frame[data_type][dataset]

            for dataset, data in outputs['test'].items():
                eval_output[dataset] = data

            return results_dict, eval_output
        else:
            return None, None

    def train(self, model, learning_rate, learning_rate_divisor, alpha, stopping_iterations, eval_steps, test=False,
              model_name=''):

        info('Resetting train loader')
        self.joint_train_loader = JointLoader(loader_dict=self.train_loaders)

        assert alpha >= 0
        assert alpha <= 1

        # Figure out skipping
        compute_met = True
        compute_wsd = True
        self.joint_train_loader.skip_none()
        if not issubclass(type(model), BaseMD) or (issubclass(type(model), BaseComplete) and alpha == 0):
            info('Training on word-sense disambiguation: skipping VUAMC')
            assert issubclass(type(model), BaseWSD)
            compute_met = False
            self.joint_train_loader.skip_met()
        if not issubclass(type(model), BaseWSD) or (issubclass(type(model), BaseComplete) and alpha == 1):
            info('Training on metaphor identification: skipping SemCor')
            assert issubclass(type(model), BaseMD)
            compute_wsd = False
            self.joint_train_loader.skip_wsd()
        assert compute_met or compute_wsd
        if compute_met and compute_wsd:
            info(f'Training jointly')
            assert issubclass(type(model), BaseComplete)
            if alpha == 0:
                self.joint_train_loader.skip_met()
            elif alpha == 1:
                self.joint_train_loader.skip_wsd()

        info('Initialising training setup')
        # self.aggregate_train_loader.set_ratios(vuamc_fraction=vuamc_fraction)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        model.train()
        best_dev_loss = math.inf

        best_wsd_f1 = 0
        best_met_f1 = 0

        stable = 0
        stopping_met = False

        info('Commencing train loop')
        for step, (met_data, wsd_data) in enumerate(self.joint_train_loader):

            optimizer.zero_grad()

            # Met loss
            if compute_met:
                (_, token_embs, sentence_embs, word_ids, wsd_opts_hot, wsd_opts_inds, mpd_opts_hot, mpd_opts_inds, labels, precomputed_wsd) = met_data
                met_probabilities = model.forward_metaphor(token_embs, sentence_embs, word_ids, wsd_opts_hot,
                                                           wsd_opts_inds, mpd_opts_hot, mpd_opts_inds, precomputed_wsd)
                met_loss = self.met_criterion(met_probabilities, labels.float()).mean(dim=0)

            # WSD loss
            if compute_wsd:
                (_, token_embs, sentence_embs, word_ids, wsd_opts_hot, wsd_opts_inds, _, _, labels, precomputed_wsd) = wsd_data
                sense_probabilities = model.forward_senses(token_embs, sentence_embs, word_ids, wsd_opts_hot,
                                                           wsd_opts_inds, precomputed_wsd)
                wsd_loss = self.sense_criterion(sense_probabilities, labels).mean(dim=0)

            if compute_met and not compute_wsd:
                loss = met_loss
            elif compute_wsd and not compute_met:
                loss = wsd_loss
            else:
                loss = alpha * met_loss + (1 - alpha) * wsd_loss

            # Process
            loss.backward()
            optimizer.step()

            if step % eval_steps == 0 or test:
                model.eval()

                # train_results = self.overall_results(model, self.train_loaders, train_ratios)
                dev_results, _ = self.overall_results(model, self.dev_loaders)

                # train_results.columns.name = 'TRAIN'
                dev_results.columns.name = 'DEV'

                # Track stable iterations
                if compute_met and not compute_wsd:
                    dev_loss = dev_results['loss']['vuamc']
                elif compute_wsd and not compute_met:
                    dev_loss = dev_results['loss']['semcor']
                else:
                    dev_loss = alpha * dev_results['loss']['vuamc'] + \
                               (1 - alpha) * dev_results['loss']['semcor']

                if improved(best_so_far=best_dev_loss, new_result=dev_loss, metric='loss'):  # and (step == 0 or not test):
                        # If it is test, don't reset stable count
                    stable = 0
                    best_dev_loss = dev_loss
                    model.set_best()
                    best_wsd_f1 = dev_results['f1']['semcor']
                    best_met_f1 = dev_results['f1']['vuamc']
                else:
                    stable += 1

                # Logging
                info(
                    f"""{model_name} training update:\n{dev_results}\n{stable}/{stopping_iterations} stable iterations; best dev loss {best_dev_loss:e} achieved Met/WSD f1s of {best_met_f1:.4f}/{best_wsd_f1:.4f}""")

                # Early stopping
                if stable >= stopping_iterations:  # Stop early if it is test mode
                    # Stagnant
                    info('Early stopping criteria met')

                    if type(model) == SerialModel and not stopping_met:  # and compute_wsd:
                        info("Continuing training only metaphor")
                        stopping_met = True
                        compute_wsd = False
                        compute_met = True  # This will usually be true anyway, unless it was just training WSD before
                        stable = 0
                        alpha = 1
                        self.joint_train_loader.skip_none()
                        self.joint_train_loader.skip_wsd()

                        model.recover_best()
                        model.freeze_wsd_model()

                        # Reset optimizer to just have the MPD parameters
                        optimizer = torch.optim.AdamW(model.sense_met_model.parameters(),
                                                      lr=learning_rate/learning_rate_divisor)

                        # Compute best metaphor only loss
                        dev_results, _ = self.overall_results(model, self.dev_loaders)

                        # assert dev_results['f1']['semcor'] == best_wsd_f1
                        assert dev_results['f1']['vuamc'] == best_met_f1

                        dev_loss = dev_results['loss']['vuamc']
                        best_dev_loss = dev_loss

                        info(f'New best loss {best_dev_loss:e}; recovered results:\n{dev_results}')
                        # NB these results are correct, and so is the assertions.

                    else:
                        info('Terminating')
                        break

                # Time quit
                if not self.time_remaining():
                    info('Out of Time')
                    return False

                model.train()

        info('Recovering best model and returning')
        model.recover_best()
        model.eval()
        return True

    def overall_results(self, model, loaders):

        results = dict()
        output = dict()

        with torch.no_grad():

            for dataset, loader in loaders.items():

                if (issubclass(type(model), BaseMD) and dataset == 'vuamc') or \
                        (issubclass(type(model), BaseWSD) and dataset == 'semcor' and
                         ((not isinstance(model.wsd_model, WSD_Precomputed)) if isinstance(model, SerialModel) else True)):

                    # Reset lists
                    all_datapoint_ids = []
                    all_predictions = []
                    all_labels = []
                    all_losses = []

                    if dataset == 'vuamc':
                        met_mode = True
                        average_mode = 'binary'
                    else:
                        assert dataset == 'semcor'
                        met_mode = False
                        average_mode = 'micro'

                    for (datapoint_ids, token_embs, sentence_embs, word_ids, wsd_opts_hot, wsd_opts_inds, mpd_opts_hot,
                         mpd_opts_inds, labels, precomputed_wsd) in loader:

                        all_datapoint_ids += datapoint_ids

                        if met_mode:
                            log_probabilities = model.forward_metaphor(token_embs, sentence_embs, word_ids, wsd_opts_hot,
                                                                       wsd_opts_inds, mpd_opts_hot, mpd_opts_inds, precomputed_wsd)
                            all_losses += self.met_criterion(log_probabilities, labels.float()).tolist()
                            all_predictions += [round(probability) for probability in
                                                torch.exp(log_probabilities).tolist()]
                        else:
                            log_probabilities = model.forward_senses(token_embs, sentence_embs, word_ids, wsd_opts_hot,
                                                                     wsd_opts_inds, precomputed_wsd)
                            all_losses += self.sense_criterion(log_probabilities, labels).tolist()
                            all_predictions += torch.argmax(log_probabilities, dim=-1).tolist()
                        all_labels += labels.tolist()

                    # Calc final values
                    loss = np.mean(all_losses)
                    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_predictions,
                                                                       average=average_mode)
                    acc = accuracy_score(all_labels, all_predictions)

                    output[dataset] = [('datapoint_ids', 'predictions')] + list(zip(all_datapoint_ids, all_predictions))

                else:
                    # Stub; not trained
                    loss = None
                    f1 = None
                    acc = None
                    rec = None
                    prec = None

                    output[dataset] = []

                # Store values
                result_dict = dict()
                result_dict['loss'] = loss
                result_dict['f1'] = f1
                result_dict['accuracy'] = acc
                result_dict['recall'] = rec
                result_dict['precision'] = prec

                results[dataset] = result_dict

        results_dataframe = pd.DataFrame.from_dict(results, orient='index')

        return results_dataframe, output

    def positive_training_time(self):
        now = time.time()
        return self.deadline - now > 0  # Return true if time left over

    def predict(self, model, loaders):

        # Return stubs if model is not type
        compute_predictions = issubclass(type(model), BaseMPD)

        results = dict()
        full_output = dict()

        with torch.no_grad():

            for dataset, loader in loaders.items():

                if compute_predictions:
                    all_datapoints = []
                    all_probabilities = []
                    all_labels = []

                    # Compute all probabilities by iterating through loader
                    for (word_ids, datapoint_ids, synsets_hot, synset_indices, labels) in loader:

                        metaphoricity_log_probs_hot = model.forward_wordnet(word_ids, synsets_hot, synset_indices)
                        metaphoricity_probs_hot = torch.exp(metaphoricity_log_probs_hot)

                        # Flatten and extract
                        batch_size, max_options = synset_indices.shape
                        batch_indices_flat = torch.tensor([[i] * max_options for i in range(batch_size)]) \
                            .reshape(batch_size * max_options)
                        synset_indices_flat = synset_indices.reshape(batch_size * max_options)
                        selected_probabilities = metaphoricity_probs_hot[
                            batch_indices_flat, synset_indices_flat].reshape(batch_size, max_options)

                        for i, num_datapoints in enumerate([len(d) for d in datapoint_ids]):
                            all_probabilities += [selected_probabilities[i, :num_datapoints].tolist()]
                        all_datapoints += datapoint_ids
                        all_labels += labels
                else:
                    all_datapoints = []
                    all_labels = []
                    all_probabilities = []

                result_dict, output = evaluate(all_datapoints, all_probabilities, all_labels)

                full_output[dataset] = output
                results[dataset] = result_dict

        results_dataframe = pd.DataFrame.from_dict(results, orient='index')

        return results_dataframe, full_output
