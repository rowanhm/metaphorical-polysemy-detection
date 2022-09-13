# p(m | w, s)
# Computes metaphoricity using a word embedding and a sense embedding
import torch
from torch.nn.functional import logsigmoid

from src.models.base import BaseMPD
from src.models.general.mlp import MLP
from src.shared.global_variables import device, seed
torch.manual_seed(seed)


class MPD_Baseline(BaseMPD):

    def __init__(self, dropout, n_layers, word_emb_tensor, sense_emb_tensor, hidden_size):
        super().__init__()

        sense_emb_size = sense_emb_tensor.shape[1]
        word_emb_size = word_emb_tensor.shape[1]

        self.word_embs = word_emb_tensor
        self.sense_embs = sense_emb_tensor
        assert not self.word_embs.requires_grad
        assert not self.sense_embs.requires_grad

        self.mlp = MLP(input_size=word_emb_size + sense_emb_size, hidden_size=hidden_size, output_size=1, dropout=dropout,
                       n_layers=n_layers)

    def forward_wordnet(self, word_ids, options_hot, options_inds):

        num_options = options_inds.shape[1]
        batch_size, sense_inventory_size = options_hot.shape

        # Get embeddings
        sense_embs = self.sense_embs[options_inds].to(device=device)  # batch_size * max_sense_opts * emb_size
        word_embs = self.word_embs[word_ids].to(device=device)

        # Concat and process to log probs
        concat = torch.cat([sense_embs, word_embs.unsqueeze(1).expand(-1, num_options, -1)], dim=-1)
        mlp_output = self.mlp(concat)
        log_prob = logsigmoid(mlp_output).squeeze(-1)

        # Reshape - flatten the indices and data
        log_prob_flat = torch.flatten(log_prob)
        batch_inds = torch.tensor(range(batch_size)).unsqueeze(-1).expand(-1, num_options).to(device=device)
        batch_inds_flat = torch.flatten(batch_inds)
        options_inds_flat = torch.flatten(options_inds)

        # Put the log probs at the right indices
        log_prob_hot = torch.tensor([float('-inf')]).repeat([batch_size, sense_inventory_size]).to(device=device)
        log_prob_hot[batch_inds_flat, options_inds_flat] = log_prob_flat
        log_prob_hot[:, 0] = float('-inf')  # reset the mask to -inf

        assert ((log_prob_hot != float('-inf')) == options_hot).all()
        return log_prob_hot
