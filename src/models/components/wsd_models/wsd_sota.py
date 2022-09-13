# p(s | w, c)
# Reimplementation of EWISER, for (1) use in p(m | w, c), (2) comparison to p(m, s | w, c)
import torch
from torch import log_softmax

from src.models.base import BaseWSD
from src.models.general.mlp import MLP
from src.shared.global_variables import seed, device

torch.manual_seed(seed)


class WSD_EWISER(BaseWSD):

    def __init__(self, token_emb_size, sense_emb_tensor, dropout, n_layers, adjacency_matrix_tensor, hidden_size):
        super().__init__()

        self.sense_inventory_size = sense_emb_tensor.shape[0]

        self.mlp = MLP(input_size=token_emb_size, output_size=token_emb_size, dropout=dropout, n_layers=n_layers,
                       hidden_size=hidden_size)

        self.a = adjacency_matrix_tensor.to(device=device)
        self.o = sense_emb_tensor.to(device=device).t()  # emb_dim x num_senses

        assert not self.a.requires_grad
        assert not self.o.requires_grad

    def forward_senses(self, token_embs, sentence_embs, word_ids, options_hot, options_inds, precomputed_wsd):

        h = self.mlp(token_embs)
        z = torch.mm(h, self.o)
        q = torch.mm(self.a, z.t()).t() + z

        log_probs = log_softmax(q, dim=-1)
        selected_log_probabilities = self.select_sense_subset(log_probs, options_hot)
        assert ((selected_log_probabilities != float('-inf')) == options_hot).all()

        return selected_log_probabilities

    def select_sense_subset(self, log_probabilities, options_one_hot):
        batch_size = log_probabilities.shape[0]

        negative_infs = torch.tensor([float('-inf')]).expand([batch_size, self.sense_inventory_size]).to(device=device)
        selected_log_probs = torch.where(options_one_hot, log_probabilities, negative_infs)

        log_sums = torch.logsumexp(selected_log_probs, dim=-1)

        renormalised = torch.where(options_one_hot,
                                   log_probabilities - log_sums.unsqueeze(-1).expand(-1, self.sense_inventory_size),
                                   negative_infs)
        return renormalised