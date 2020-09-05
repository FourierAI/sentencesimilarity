#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: bert_client.py
# @time: 2020-07-26 12:06
# @desc:

from bert_serving.client import BertClient
import numpy as np


class MyBertClient():
    def __init__(self):
        self.bc = BertClient()

    def query_sentence_vec(self, sentence):
        return self.bc.encode([sentence])[0][1:-1]

    def query_sentences_vec(self, sentences):
        return self.bc.encode(sentences)

    def query_simility_sentence_pair(self, sent1, sent2):
        pair_vec = self.bc.encode([sent1, sent2])
        sent1_vec = pair_vec[0]
        sent2_vec = pair_vec[1]

        sent1_vec_len = np.linalg.norm(sent1_vec)
        sent2_vec_len = np.linalg.norm(sent2_vec)

        score = sent1_vec.dot(sent2_vec) / (sent1_vec_len * sent2_vec_len)

        return score.tolist()


if __name__ == "__main__":

    from dataprocess.bert_pretrain.bert_service import MyBertServer

    bs = MyBertServer()
    bc = MyBertClient()
    sent_vec = bc.query_sentence_vec('I love you')
    print(sent_vec)
