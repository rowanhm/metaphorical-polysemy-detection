from math import ceil
import torch
from nltk.corpus import wordnet as wn
from transformers import BertTokenizerFast, BertModel

from src.shared.common import open_pickle, save_pickle, info, chunks
from src.shared.global_variables import bert_model, device, preprocessing_batch_size, embedding_mode, \
    combo_mode, md_vuamc_vocab_file, wsd_semcor_vocab_file, word_ids_file, word_embs_file, seed, mpd_mml_vocab_file
from src.preprocessing.helper import filter_lemmas

torch.manual_seed(seed)


def build_word_tensor():
    info(f"Initialising BERT using device {device}")
    tokenizer = BertTokenizerFast.from_pretrained(bert_model)
    bert = BertModel.from_pretrained(bert_model).to(device=device)
    bert.eval()
    torch.no_grad()

    info('Getting vocab from VUAMC and SemCor')
    vuamc_vocab = open_pickle(md_vuamc_vocab_file)
    semcor_vocab = open_pickle(wsd_semcor_vocab_file)

    all_vocab = vuamc_vocab.union(semcor_vocab)
    info(f'{len(vuamc_vocab)} in VUAMC, {len(semcor_vocab)} in SemCor, {len(all_vocab)} combined')

    info('Adding words from MML subset')
    wn_subset_vocab = open_pickle(mpd_mml_vocab_file)
    all_vocab = all_vocab.union(wn_subset_vocab)
    info(f'{len(wn_subset_vocab)} in subset, now total is {len(all_vocab)}')

    info('Adding words from annotation subset')
    all_vocab = all_vocab.union(wn_subset_vocab)
    info(f'{len(wn_subset_vocab)} in subset, now total is {len(all_vocab)}')

    info('Adding words from WordNet')
    wn_vocab = set()
    for synset in wn.all_synsets():
        lemmas = synset.lemmas()
        for lemma in filter_lemmas(lemmas):
            # If the lemmas has multiple associated synsets
            if len(wn.synsets(lemma.name())) > 1:
                # word = ' '.join(lemma.name().split('_'))
                wn_vocab.add(lemma.name())
    all_vocab = all_vocab.union(wn_vocab)
    info(f'{len(wn_vocab)} in WordNet, so {len(all_vocab)} items in total combined vocab')

    info('Building word dict and tokenizing')
    vocab_bert_tokens = []
    vocab_dict = dict()
    for i, word in enumerate(all_vocab):
        vocab_dict[word] = i+1  # Add one to leave room for mask
        word_spaced = ' '.join(word.split('_'))

        token_data = tokenizer(word_spaced)['input_ids']
        vocab_bert_tokens += [(i+1, token_data)]

    info('Embedding')
    vocab_bert_embeddings = []
    batches = chunks(vocab_bert_tokens, preprocessing_batch_size)
    running_count = 0
    for batch_id, batch in enumerate(batches):

        if (batch_id + 1) % 50 == 0:
            info(f"On batch {batch_id + 1}/{ceil(len(vocab_bert_tokens) / preprocessing_batch_size)}")

        ids, bert_tokens = zip(*batch)

        # Make sure the correct order is maintained
        for id in ids:
            assert id == running_count+1
            running_count += 1

        # Build input tensor
        tensor_shape = [len(bert_tokens), max([len(tokens) for tokens in bert_tokens])]
        tokens_tensor = torch.zeros(tensor_shape).long().to(device=device)
        attention_tensor = torch.zeros(tensor_shape).bool().to(device=device)

        token_lengths = []
        for i, tokens in enumerate(bert_tokens):
            length = len(tokens)

            tokens_tensor[i, :length] = torch.tensor(tokens)
            attention_tensor[i, :length] = torch.tensor([1] * length)

            token_lengths += [length]

        output = bert(tokens_tensor, attention_mask=attention_tensor, output_hidden_states=True)[2]

        # Choose which layer to take
        if embedding_mode == 'take-last':
            bert_embs_batch = output[-1].detach()
        else:
            assert embedding_mode == 'average-4'
            bert_embs_batch = output[-1] + output[-2] + output[-3] + output[-4]
            bert_embs_batch = torch.div(bert_embs_batch, 4).detach()

        for i, (embeddings, length) in enumerate(zip(bert_embs_batch, token_lengths)):

            # Choose how to combine
            if combo_mode == 'take-first':
                token_emb = embeddings[1]
            else:
                assert combo_mode == 'average'
                token_emb = torch.mean(embeddings[1:length-1], dim=0)

            vocab_bert_embeddings += [token_emb.cpu()]

    info('Building final tensor')
    vocab_bert_embeddings = [torch.zeros([vocab_bert_embeddings[0].shape[0]])] + vocab_bert_embeddings  # Add the mask
    vocab_embeddings = torch.stack(vocab_bert_embeddings).cpu()

    info('Saving')
    save_pickle(word_ids_file, vocab_dict)
    save_pickle(word_embs_file, vocab_embeddings)

    info('Done')


if __name__ == "__main__":
    build_word_tensor()
