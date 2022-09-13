import torch.utils.data
import torch.nn as nn

from src.shared.global_variables import seed
torch.manual_seed(seed)


class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout):
        super().__init__()

        assert n_layers > 0

        self.input_size = input_size
        self.n_layers = n_layers

        if n_layers == 1:

            # If it has one layer, it is simply a linear layer
            mlp = [nn.Dropout(dropout), nn.Linear(input_size, output_size)]

        else:

            mlp = [nn.Dropout(dropout), nn.Linear(input_size, hidden_size)]

            # Middle layers
            for i in range(1, n_layers-1):
                mlp += [nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size, hidden_size)]

            # Last layer
            mlp += [nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, output_size)]

        self.mlp = nn.Sequential(*mlp)

    def forward(self, embedding):

        return self.mlp(embedding)
