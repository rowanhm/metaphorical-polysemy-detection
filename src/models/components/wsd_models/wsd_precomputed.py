import torch

from src.shared.global_variables import seed, device
from src.models.base import BaseWSD
torch.manual_seed(seed)


class WSD_Precomputed(BaseWSD):

    def __init__(self):
        super().__init__()

    def forward_senses(self, token_embs, sentence_embs, word_ids, options_hot, options_inds, precomputed_wsd):

        log_probs = torch.full(size=options_hot.shape, fill_value=float('-inf'))
        for batch_id, (indices, values) in enumerate(precomputed_wsd):
            log_probs[batch_id, indices] = torch.log(torch.tensor(values))

        log_probs = log_probs.to(device=device)
        return log_probs
