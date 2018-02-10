from collections import Counter
from gensim.models import Word2Vec
from random import random
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from torch import nn
from torch.autograd import Variable

import numpy as np
import torch
import torch.nn.functional as F

# Load the data into memory.
train_sentences = [line.strip() for line in open("data3/mscoco_train_captions.txt").readlines()]
val_sentences = [line.strip() for line in open("data3/mscoco_val_captions.txt").readlines()]

train_sentences = [x for x in train_sentences if x] 
val_sentences = [x for x in val_sentences if x] 
print(len(train_sentences))
print(len(val_sentences))
print(train_sentences[0])
print(train_sentences[1])
print(train_sentences[2])

sentences = train_sentences

# Lower-case the sentence, tokenize them and add <SOS> and <EOS> tokens
sentences = [["<SOS>"] + word_tokenize(sentence.lower()) + ["<EOS>"] for sentence in sentences]

# Create the vocabulary. Note that we add an <UNK> token to represent words not in our vocabulary.
vocabularySize = 1000
word_counts = Counter([word for sentence in sentences for word in sentence])
vocabulary = ["<UNK>"] + [e[0] for e in word_counts.most_common(vocabularySize-1)]
word2index = {word:index for index,word in enumerate(vocabulary)}
one_hot_embeddings = np.eye(vocabularySize)

# Build the word2vec embeddings
wordEncodingSize = 300
filtered_sentences = [[word for word in sentence if word in word2index] for sentence in sentences]
w2v = Word2Vec(filtered_sentences, min_count=0, size=wordEncodingSize)
w2v_embeddings = np.concatenate((np.zeros((1, wordEncodingSize)), w2v.wv.syn0))

# Define the max sequence length to be the longest sentence in the training data. 
maxSequenceLength = max([len(sentence) for sentence in sentences])

def numberize(sentence):
    numberized = [word2index.get(word, 0) for word in sentence]
    return numberized

def one_hot(sentence):
    numberized = numberize(sentence)
    # Represent each word as it's one-hot embedding
    one_hot_embedded = one_hot_embeddings[numberized]
    
    return one_hot_embedded


print(sentences[1])
print(numberize(sentences[1]))
print(one_hot(sentences[1]))


