#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: evaluate_model.py
# @time: 2020-07-10 19:12
# @desc:

import sys

sys.path.append('../')
import prj_model.LSTM as LSTM
import torch
import dataprocess.embeddingword as embeddingword
import dataprocess.loaddata as ld
import prj_config.embeddingconfig as embeddingconfig

if __name__ == "__main__":
    embedding_config = embeddingconfig.EmbeddingConfig()

    scores = ld.load_score(data_set='sts', type='test')
    scores = [score / 5 for score in scores]

    test_sentences_left, test_sentences_right = embeddingword.sent2tensors()
    rnn = torch.load('../train/rnn.pth')

    for step, left_sent in enumerate(test_sentences_left):
        right_sent = test_sentences_right[step]
        left_encoded_vec = rnn(left_sent.view(1, left_sent.shape[0], embedding_config.dimension))
        right_encoded_vec = rnn(right_sent.view(1, right_sent.shape[0], embedding_config.dimension))

        left_encoded_vec = left_encoded_vec.squeeze()
        left_encoded_vec_mod = torch.sqrt(torch.sum(left_encoded_vec * left_encoded_vec))

        right_encoded_vec = right_encoded_vec.squeeze()
        right_encoded_vec_mod = torch.sqrt(torch.sum(right_encoded_vec * right_encoded_vec))

        cos_simility = sum(left_encoded_vec * right_encoded_vec) / left_encoded_vec_mod / right_encoded_vec_mod

        print('computation simility:{} and target simility:{}.'.format(cos_simility, scores[step]))
