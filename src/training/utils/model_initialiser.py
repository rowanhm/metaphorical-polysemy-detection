from src.models.components.wsd_models.wsd_precomputed import WSD_Precomputed
from src.shared.common import info
from src.training.parameters.parameters import ParameterKeys
from src.shared.global_variables import device


def initialise_model(params, token_emb_size, word_emb_tensor, sense_emb_tensor, contexts_dict, sense_adjacency_tensor):

    model_name = params[ParameterKeys.MODEL.name]
    dropout = params[ParameterKeys.DROPOUT.name]
    n_layers_1 = params[ParameterKeys.N_LAYERS_1.name]
    n_layers_2 = params[ParameterKeys.N_LAYERS_2.name]
    hidden_size_1 = params[ParameterKeys.HIDDEN_SIZE_1.name]
    hidden_size_2 = params[ParameterKeys.HIDDEN_SIZE_2.name]
    # gnn_type = params[ParameterKeys.GNN_TYPE.name]

    word_emb_size = word_emb_tensor.shape[1]
    sense_inventory_size, sense_emb_size = sense_emb_tensor.shape

    sentence_emb_size = token_emb_size

    info(f'Initialising {model_name} [token_emb_size={token_emb_size}; word_emb_size={word_emb_size}; sense_emb_size={sense_emb_size}]')

    model_name = model_name.split('.')
    model_type = model_name[0]

    if model_type == 'serial':

        wsd_implementation = model_name[1]
        sense_met_implementation = model_name[2]

        # Init p(s | w, c)
        if wsd_implementation == 'baseline':
            from src.models.components.wsd_models.wsd_baseline import WSD_Baseline
            wsd_model = WSD_Baseline(token_emb_size=token_emb_size, sense_inventory_size=sense_inventory_size,
                                     dropout=dropout, n_layers=n_layers_1, hidden_size=hidden_size_1)
        else:
            assert wsd_implementation == 'sota'
            from src.models.components.wsd_models.wsd_sota import WSD_EWISER
            wsd_model = WSD_EWISER(token_emb_size=token_emb_size, sense_emb_tensor=sense_emb_tensor, dropout=dropout,
                                   n_layers=n_layers_1, adjacency_matrix_tensor=sense_adjacency_tensor,
                                   hidden_size=hidden_size_1)


        # Init p(m | w, s)
        if sense_met_implementation == 'baseline':
            from src.models.components.mpd_models.mpd_baseline import MPD_Baseline
            sense_met_model = MPD_Baseline(dropout=dropout, n_layers=n_layers_2, word_emb_tensor=word_emb_tensor,
                                           sense_emb_tensor=sense_emb_tensor, hidden_size=hidden_size_2)
        else:
            assert False

        from src.models.wrappers.complete_serial import SerialModel
        model = SerialModel(sense_met_model=sense_met_model, wsd_model=wsd_model)

    elif model_type == 'met':
        implementation = model_name[1]

        if implementation == 'baseline':
            from src.models.components.md_models.md_baseline import MD_Baseline
            model = MD_Baseline(token_emb_size=token_emb_size, dropout=dropout, n_layers=n_layers_1,
                                hidden_size=hidden_size_1)
        else:
            assert implementation == 'sota'
            from src.models.components.md_models.md_sota import MD_MelBERT
            model = MD_MelBERT(n_layers_spv=n_layers_1, n_layers_mip=n_layers_2, dropout=dropout,
                               sentence_emb_size=sentence_emb_size, token_emb_size=token_emb_size,
                               word_emb_tensor=word_emb_tensor, hidden_size_spv=hidden_size_1,
                               hidden_size_mip=hidden_size_2)

    else:
        assert model_type == 'wsd'
        implementation = model_name[1]

        if implementation == 'baseline':
            from src.models.components.wsd_models.wsd_baseline import WSD_Baseline
            model = WSD_Baseline(token_emb_size=token_emb_size, sense_inventory_size=sense_inventory_size,
                                 dropout=dropout, n_layers=n_layers_1, hidden_size=hidden_size_2)
        else:
            assert implementation == 'sota'
            from src.models.components.wsd_models.wsd_sota import WSD_EWISER
            model = WSD_EWISER(token_emb_size=token_emb_size, sense_emb_tensor=sense_emb_tensor, dropout=dropout,
                               n_layers=n_layers_1, adjacency_matrix_tensor=sense_adjacency_tensor,
                               hidden_size=hidden_size_2)

    info(f'Moving model to {device}')
    model = model.to(device=device)

    info('Setting model best')
    model.set_best()

    return model
