import random

from src.shared.global_variables import seed

random.seed(seed)


class JointLoader:

    def __init__(self, loader_dict):

        self.loaders = loader_dict
        # Build iterator dictionary
        self.iters = dict()
        for dataset, loader in self.loaders.items():
            self.iters[dataset] = iter(loader)

        self.skip_none()

    def __iter__(self):
        return self

    def __next__(self):

        # Call next
        met_data = self.get_met()
        wsd_data = self.get_wsd()

        return met_data, wsd_data

    def get_data(self, dataset):
        data = next(self.iters[dataset], None)

        if data is None:
            self.reset_iter(dataset)
            data = next(self.iters[dataset])

        return data

    def reset_iter(self, dataset):
        self.iters[dataset] = iter(self.loaders[dataset])

    def skip_met(self):
        self.get_met = lambda: None

    def skip_wsd(self):
        self.get_wsd = lambda: None

    def skip_none(self):
        self.get_met = lambda: self.get_data('vuamc')
        self.get_wsd = lambda: self.get_data('semcor')

