import abc
import copy

from torch import nn


class Base(nn.Module):

    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)


class BaseWSD(Base, abc.ABC):

    @abc.abstractmethod
    def forward_senses(self, token_embs, sentence_embs, word_ids, wsd_opts_hot, wsd_opts_inds, precomputed_wsd):
        pass


class BaseMD(Base, abc.ABC):

    @abc.abstractmethod
    def forward_metaphor(self, token_embs, sentence_embs, word_ids, wsd_opts_hot, wsd_opts_inds, mpd_opts_hot,
                         mpd_opts_inds, precomputed_wsd):
        pass


class BaseMPD(Base, abc.ABC):

    @abc.abstractmethod
    def forward_wordnet(self, word_ids, options_hot, options_inds):
        pass


class BaseJoint(BaseWSD, BaseMD, abc.ABC):

    pass


class BaseComplete(BaseWSD, BaseMD, BaseMPD, abc.ABC):

    pass

