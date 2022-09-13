# p(m | w, c)
# Reimplementation of MelBERT, for (1) comparison to p(m, s | w, c), and (2) to p(m | w, c) which uses sense addition
import torch
from torch import nn
from torch.nn.functional import logsigmoid

from src.models.general.mlp import MLP
from src.models.base import BaseMD
from src.shared.global_variables import seed, device

torch.manual_seed(seed)


class MD_MelBERT(BaseMD):

    def __init__(self, n_layers_spv, n_layers_mip, dropout, sentence_emb_size, token_emb_size, word_emb_tensor,
                 hidden_size_spv, hidden_size_mip):
        super().__init__()

        word_emb_size = word_emb_tensor.shape[1]

        self.spv = MLP(n_layers=n_layers_spv, dropout=dropout, input_size=(sentence_emb_size+token_emb_size),
                       hidden_size=hidden_size_spv, output_size=token_emb_size)
        self.mip = MLP(n_layers=n_layers_mip, dropout=dropout, input_size=(word_emb_size+token_emb_size),
                       hidden_size=hidden_size_mip, output_size=token_emb_size)
        self.final = nn.Linear((token_emb_size*2), 1)

        self.word_embs = word_emb_tensor
        assert not self.word_embs.requires_grad

    def forward_metaphor(self, token_embs, sentence_embs, word_ids, wsd_opts_hot, wsd_opts_inds, mpd_opts_hot,
                         mpd_opts_inds, precomputed_wsd):

        spv = self.spv(torch.cat((token_embs, sentence_embs), dim=-1))

        word_embs = self.word_embs[word_ids].to(device=device)
        mip = self.spv(torch.cat((token_embs, word_embs), dim=-1))

        output = self.final(torch.cat((spv, mip), dim=-1)).squeeze(-1)
        log_probs = logsigmoid(output)
        return log_probs

