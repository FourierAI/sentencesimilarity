#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: embeddingword.py
# @time: 2020-07-06 11:54
# @desc:

import dataprocess.loaddata as ld
from gensim.models import Word2Vec
from prj_config.embeddingconfig import EmbeddingConfig
from prj_config.global_config import GlobalConfig
from nltk.tokenize import word_tokenize
import torch
import pickle


# pre_train word2vec
def pre_train_word2vec(sents, embedding_config):
    model = Word2Vec(sents, sg=embedding_config.Word2Vec_model, size=embedding_config.dimension,
                     window=embedding_config.Word2Vec_WindowSize, min_count=embedding_config.Word2Vec_minicount,
                     workers=embedding_config.Word2Vec_Worker)
    model.save('Word2Vec.prj_model')
    return model


def sent2tensor(sentences, word2vec_or_embedding, word2vec_model=None, word_embedding=None, word_dict=None):
    sents_embedding = []
    sents = [word_tokenize(sent[0].lower()) for sent in sentences]
    for sent in sents:
        sent_embedding = []
        for word in sent:
            if word2vec_or_embedding == 'word2vec':
                sent_embedding.append(word2vec_model.wv.__getitem__(word).tolist()[0])
            else:
                sent_embedding.append(word_embedding(torch.LongTensor([word_dict[word]])).tolist()[0])
        sent_embedding_tensor = torch.FloatTensor(sent_embedding)
        sents_embedding.append(sent_embedding_tensor)
    return sents_embedding


def sent2tensors():
    global_config = GlobalConfig()
    embedding_config = EmbeddingConfig()
    sents_embedding_left = []
    sents_embedding_right = []
    if global_config.train_or_test == "train":

        train_sents_left, train_sents_right = ld.load_sents_pair(global_config.data_set, type='train')
        whole_sents = ld.load_all_sents(global_config.data_set)

        if embedding_config.pre_train:
            word2vec_model = pre_train_word2vec(whole_sents, embedding_config)
            sents_embedding_left = sent2tensor(sentences=train_sents_left, word2vec_or_embedding='word2vec',
                                               word2vec_model=word2vec_model)
            sents_embedding_right = sent2tensor(sentences=train_sents_right, word2vec_or_embedding='word2vec',
                                                word2vec_model=word2vec_model)
        else:
            word_set, word_dict = get_wordset_and_worddict(whole_sents)
            word_capability = len(word_set)
            word_embedding = torch.nn.Embedding(word_capability, embedding_config.dimension)
            with open('word_dict.prj_model', 'wb') as file:
                pickle.dump(word_dict, file)
            with open('word_embedding.prj_model', 'wb') as file:
                pickle.dump(word_embedding, file)

            sents_embedding_left = sent2tensor(sentences=train_sents_left, word2vec_or_embedding='embedding',
                                               word_embedding=word_embedding, word_dict=word_dict)
            sents_embedding_right = sent2tensor(sentences=train_sents_right, word2vec_or_embedding='embedding',
                                                word_embedding=word_embedding, word_dict=word_dict)
    elif global_config.train_or_test == "test":
        test_sents_left, test_sents_right = ld.load_sents_pair(global_config.data_set, global_config.train_or_test)
        if embedding_config.pre_train == True:
            word2vec_model = Word2Vec.load('Word2Vec.prj_model')
            sents_embedding_left = sent2tensor(sentences=test_sents_left, word2vec_or_embedding='word2vec',
                                               word2vec_model=word2vec_model)
            sents_embedding_right = sent2tensor(sentences=test_sents_right, word2vec_or_embedding='word2vec',
                                                word2vec_model=word2vec_model)
        else:
            with open('word_dict.prj_model', 'rb') as file:
                word_dict = pickle.load(file)

            with open('word_embedding.prj_model', 'rb') as file:
                word_embedding = pickle.load(file)

            sents_embedding_left = sent2tensor(sentences=test_sents_left, word2vec_or_embedding='embedding',
                                               word_embedding=word_embedding, word_dict=word_dict)
            sents_embedding_right = sent2tensor(sentences=test_sents_right, word2vec_or_embedding='embedding',
                                                word_embedding=word_embedding, word_dict=word_dict)

    return sents_embedding_left, sents_embedding_right


def get_wordset_and_worddict(whole_sents):
    word_set = set()
    word_dict = {}
    index = 0
    for sent in whole_sents:
        for word in sent:
            if word not in word_set:
                word_dict[word] = index
                word_set.add(word)
                index += 1
    return word_set, word_dict


if __name__ == "__main__":
    train_sents_embedding_left, train_sents_embedding_right = sent2tensors()

    print(train_sents_embedding_left)

    print(train_sents_embedding_right)
