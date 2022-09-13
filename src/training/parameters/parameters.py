from enum import Enum, auto


class ParameterKeys(Enum):
    MODEL = auto()
    DROPOUT = auto()
    N_LAYERS_1 = auto()
    N_LAYERS_2 = auto()
    HIDDEN_SIZE_1 = auto()
    HIDDEN_SIZE_2 = auto()
    # GNN_TYPE = auto()
    LEARNING_RATE = auto()
    LEARNING_RATE_DIVISOR = auto()
    ALPHA = auto()
