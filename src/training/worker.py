from src.training.data_loaders.infinite_loader import InfiniteLoader
from src.shared.common import open_pickle, info, warn
from src.shared.global_variables import md_vuamc_train_dir, wsd_semcor_train_dir, md_vuamc_dev_file, \
    wsd_semcor_dev_file, \
    md_vuamc_test_file, wsd_semcor_test_file, train_batch_size, stopping_iterations, eval_steps, \
    word_embs_file, sense_embs_file, device, mpd_verbs_machine_file, mpd_mml_machine_file, sense_adjacency_file, \
    mpd_oov_machine_file, runtime
from src.training.utils.file_editing import FileEditor
from src.training.utils.model_initialiser import initialise_model
from src.training.utils.trainer_utils import make_static_train_loader, make_static_eval_loader, train_collate_fn
from src.training.parameters.parameters import ParameterKeys
from src.training.trainer import Trainer
from src.training.data_loaders.streaming_data_loader import StreamingDataLoader


class Worker:

    def __init__(self, runtime=None, test=False):

        self.test = test

        if not test:
            info('Loading data & initialising loaders')
            dev_data_loaders = dict({
                'vuamc': make_static_train_loader(md_vuamc_dev_file, batch_size=train_batch_size, shuffle=False),
                'semcor': make_static_train_loader(wsd_semcor_dev_file, batch_size=train_batch_size, shuffle=False)
            })
            self.train_data_loaders = dict({
                'vuamc': StreamingDataLoader(md_vuamc_train_dir, batch_size=train_batch_size, collate_fn=train_collate_fn,
                                             shuffle=True),
                'semcor': StreamingDataLoader(wsd_semcor_train_dir, batch_size=train_batch_size, collate_fn=train_collate_fn,
                                              shuffle=True)
            })
            test_data_loaders = dict({
                'vuamc': make_static_train_loader(md_vuamc_test_file, batch_size=train_batch_size, shuffle=False),
                'semcor': make_static_train_loader(wsd_semcor_test_file, batch_size=train_batch_size, shuffle=False)
            })

        else:
            warn('Worker in test mode')
            data_cap = 10
            dev_data_loaders = dict({
                'vuamc': make_static_train_loader(md_vuamc_dev_file, batch_size=train_batch_size, shuffle=False,
                                                  limit_data=data_cap),
                'semcor': make_static_train_loader(wsd_semcor_dev_file, batch_size=train_batch_size, shuffle=False,
                                                   limit_data=data_cap)
            })
            test_data_loaders = dev_data_loaders
            self.train_data_loaders = dict()
            for dataset, loader in dev_data_loaders.items():
                self.train_data_loaders[dataset] = InfiniteLoader(loader)

        eval_loaders = dict({
            'verbs': make_static_eval_loader(mpd_verbs_machine_file, batch_size=train_batch_size),
            'selection': make_static_eval_loader(mpd_mml_machine_file, batch_size=train_batch_size),
            'random': make_static_eval_loader(mpd_oov_machine_file, batch_size=train_batch_size)
        })

        self.trainer = Trainer(self.train_data_loaders, dev_data_loaders, test_data_loaders, eval_loaders, runtime=runtime)

        self.token_emb_size = next(iter(dev_data_loaders['vuamc']))[1].shape[1]  # first dataset, first datapoint, sent embedding

        info('Loading embedding tensors')
        self.word_emb_tensor = open_pickle(word_embs_file).cpu()
        self.sense_emb_tensor = open_pickle(sense_embs_file).cpu()

        info('Loading adjacency matrix')
        self.adjacency_tensor = open_pickle(sense_adjacency_file).cpu()

        # TODO remove contexts
        self.contexts = None

    def train(self, params):

        info(f"Using device {device}")
        alpha = params[ParameterKeys.ALPHA.name]
        learning_rate = params[ParameterKeys.LEARNING_RATE.name]
        learning_rate_div = params[ParameterKeys.LEARNING_RATE_DIVISOR.name]
        model_name = params[ParameterKeys.MODEL.name]
        model = initialise_model(params=params, token_emb_size=self.token_emb_size, word_emb_tensor=self.word_emb_tensor,
                                 sense_emb_tensor=self.sense_emb_tensor, contexts_dict=self.contexts,
                                 sense_adjacency_tensor=self.adjacency_tensor)

        info("Training")
        results, output = self.trainer.train_and_eval(model, alpha=alpha, learning_rate=learning_rate,
                                                      learning_rate_divisor=learning_rate_div,
                                                      stopping_iterations=stopping_iterations, eval_steps=eval_steps,
                                                      test=self.test, model_name=model_name)

        return results, output, model

    def shutdown(self):
        if not self.test:
            for name, loader in self.train_data_loaders.items():
                info(f"Terminating loader for {name}")
                loader.terminate()


def train_test():
    # info('Running test train loop')
    # worker = Worker()
    #
    # params = dict({
    #     ParameterKeys.MODEL.name: 'serial.baseline.baseline',
    #     ParameterKeys.VUAMC_FRACTION.name: 0.5,
    #     ParameterKeys.DROPOUT.name: 0.1,
    #     ParameterKeys.N_LAYERS_1.name: 1,
    #     ParameterKeys.N_LAYERS_2.name: 1
    # })
    # results, model = worker.train(params)
    # info(f'Test complete: \n{results}')
    train_from_queue(0, test=True)


def train_from_queue(experiment_id, runtime=None, test=False):

    if runtime is None:
        info(f'Training from queue {experiment_id} with no deadline')
    else:
        info(f'Training from queue {experiment_id} with a {runtime} hour deadline')

    worker = Worker(runtime, test=test)
    file_editor = FileEditor(experiment_id)

    param_id = file_editor.pop_param()

    while param_id is not None:
        info(f'Training param id {param_id}')

        params = file_editor.get_params_for_id(param_id)
        info(params)
        results, full_output, model = worker.train(params)

        if results is None:
            info('Terminating')
            file_editor.queue_param(param_id)
            worker.shutdown()
            return

        info('Done, saving results')
        file_editor.save_result(param_id, results)
        file_editor.save_model(param_id, model)
        file_editor.save_predictions(param_id, full_output)

        # Next params
        del model
        param_id = file_editor.pop_param()

    info('No parameters left - terminating')


if __name__ == "__main__":

    train_from_queue(experiment_id=0, runtime=runtime, test=True)
