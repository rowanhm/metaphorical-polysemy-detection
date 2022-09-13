from nltk.corpus import wordnet as wn

def remove_singletons(lookup):
    lookup_new = dict()
    removed = False
    for key, value in lookup.items():
        if len(value) > 1:
            lookup_new[key] = value
        else:
            removed = True
    return lookup_new

word_to_senses = dict()
sense_to_words = dict()

synsets = wn.all_synsets()

for synset in synsets:
    sense = synset.name()
    words = set(synset.lemmas())
    sense_to_words = words
    for word in words:
        word = word.name()
        if word in word_to_senses.keys():
            word_to_senses[word].add(sense)
        else:
            word_to_senses[word] = {sense}

word_to_senses = remove_singletons(word_to_senses)  # Remove any words which only reference a single sense


def senses_to_uid(senses):
    return ";".join(sorted(list(senses)))

word_count = len(word_to_senses)
word_sense_count = sum([len(senses) for senses in word_to_senses.values()])
word_sense_count_filtered = sum([len(senses) for senses in word_to_senses.values()])


sense_count = len(set.union(*word_to_senses.values()))

print('word, word-senses, senses AFTER monosemous words filtered')
print(word_count)
print(word_sense_count)
print(sense_count)

sense_sets = set([frozenset(senses) for senses in word_to_senses.values()])

print('word, word-senses AFTER words with identical sets of senses filtered')
print(len(sense_sets))
print(sum([len(senses) for senses in sense_sets]))
