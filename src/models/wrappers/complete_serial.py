import torch

from src.models.base import BaseComplete
from src.shared.global_variables import seed
torch.manual_seed(seed)

class SerialModel(BaseComplete):

    def __init__(self, sense_met_model, wsd_model):
        super().__init__()
        self.sense_met_model = sense_met_model
        self.wsd_model = wsd_model

    def forward_senses(self, token_embs, sentence_embs, word_ids, options_hot, options_inds, precomputed_wsd):
        return self.wsd_model.forward_senses(token_embs, sentence_embs, word_ids, options_hot, options_inds, precomputed_wsd)

    def forward_metaphor(self, token_embs, sentence_embs, word_ids, wsd_opts_hot, wsd_opts_inds, mpd_opts_hot,
                         mpd_opts_inds, precomputed_wsd):
        # p(m | w, c) = sum_s p(m | w, s) p(s | w, c)

        sense_log_probs = self.forward_senses(token_embs, sentence_embs, word_ids, wsd_opts_hot, wsd_opts_inds, precomputed_wsd)
        met_log_probs = self.forward_wordnet(word_ids, mpd_opts_hot, mpd_opts_inds)
        # These should be -inf in all the same places...

        product = sense_log_probs + met_log_probs  # Add logs to get the product of probs

        # assert ((product != float('-inf')) == wsd_opts_hot).all()
        assert not product.isnan().any()

        product_sum = torch.logsumexp(product, dim=-1)
        return product_sum

    def forward_wordnet(self, word_ids, options_hot, options_inds):
        return self.sense_met_model.forward_wordnet(word_ids, options_hot, options_inds)

    def freeze_wsd_model(self):
        for param in self.wsd_model.parameters():
            param.requires_grad = False
        # self.forward_sense_option = self.forward_senses_detached