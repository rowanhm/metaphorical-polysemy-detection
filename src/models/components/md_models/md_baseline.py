from torch.nn.functional import logsigmoid

from src.models.base import BaseMD
from src.models.general.mlp import MLP


class MD_Baseline(BaseMD):

    def __init__(self, token_emb_size, dropout, n_layers, hidden_size):
        super().__init__()

        self.mlp = MLP(input_size=token_emb_size, output_size=1, dropout=dropout, n_layers=n_layers,
                       hidden_size=hidden_size)

    def forward_metaphor(self, token_embs, sentence_embs, word_ids, wsd_opts_hot, wsd_opts_inds, mpd_opts_hot,
                         mpd_opts_inds, precomputed_wsd):
        # Sum over relevant senses
        output = self.mlp(token_embs).squeeze(-1)
        log_probs = logsigmoid(output)

        return log_probs
