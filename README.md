# Metahorical Polysemy Detection

## Reproducing

### Setup
Into the `data/raw` folder:
* Download and decompress the [ARES embeddings](http://sensembert.org/resources/ares_embedding.tar.gz) 
* Download and decompress the [WordNet verb annotation data](http://www.purl.org/net/metaphor-emotion-data)
* Clone the [VUAMC with novelty annotation](https://github.com/UKPLab/emnlp2018-novel-metaphors)

Run `python -m src.A_preprocess` to process the data. 
Optionally, use the `--conventional` flag to only include conventional metaphors.

### Running

Our code is designed to be run on a supercomputer.
As such, the parameter search works by initialising a queue of parameters,
which are then taken by multiple training scripts which can be run in parallel.

Run `python -m src.B_initialise` to initialise the queue of parameters to train.

Run `python -m src.C_train` to train models from the queue. You can run multiple of this script.
You can improve the speed by running it without assertions, using the `-O` flag, and adjust the runtime using the `--runtime` flag 
(default is 2 hours).

### Evaluating

To evaluate, run `python -m src.D_postprocess`. This script adds additional baselines, and evaluates them.

## Future updates

This codebase consists of code developed during experimentation, 
and as a result it is quite messy.
In the future, if there is interest, I will spend time cleaning it up, 
and will also make it possible to run the models we trained in our paper.