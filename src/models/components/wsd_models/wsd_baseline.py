import torch
from torch.nn.functional import log_softmax

from src.shared.global_variables import device, seed
from src.models.base import BaseWSD
from src.models.general.mlp import MLP
torch.manual_seed(seed)


class WSD_Baseline(BaseWSD):

    def __init__(self, token_emb_size, sense_inventory_size, dropout, n_layers, hidden_size):
        super().__init__()

        self.sense_inventory_size = sense_inventory_size

        self.mlp = MLP(input_size=token_emb_size, output_size=self.sense_inventory_size, dropout=dropout,
                       n_layers=n_layers, hidden_size=hidden_size)

    def forward_senses(self, token_embs, sentence_embs, word_ids, options_hot, options_inds, precomputed_wsd):
        all_log_probabilities = self.forward_log_probabilities(token_embs)
        selected_log_probabilities = self.select_sense_subset(all_log_probabilities, options_hot)

        assert ((selected_log_probabilities != float('-inf')) == options_hot).all()
        return selected_log_probabilities

    def forward_log_probabilities(self, token_embs):
        # Computes logits for input (outputting a logit for every <m,s> tuple)
        output = self.mlp(token_embs)
        log_probs = log_softmax(output, dim=-1)
        return log_probs

    def select_sense_subset(self, log_probabilities, options_one_hot):
        batch_size = log_probabilities.shape[0]

        negative_infs = torch.tensor([float('-inf')]).expand([batch_size, self.sense_inventory_size]).to(device=device)
        selected_log_probs = torch.where(options_one_hot, log_probabilities, negative_infs)

        log_sums = torch.logsumexp(selected_log_probs, dim=-1)

        renormalised = torch.where(options_one_hot,
                                   log_probabilities - log_sums.unsqueeze(-1).expand(-1, self.sense_inventory_size),
                                   negative_infs)
        return renormalised
