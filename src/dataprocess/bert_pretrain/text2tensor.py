#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: text2tensor.py
# @time: 2020-08-30 08:49
# @desc:
import torch
from dataprocess.bert_pretrain.bert_service import MyBertServer
from dataprocess.bert_pretrain.bert_client import MyBertClient
import dataprocess.loaddata as ld
import pandas as pd
import csv
import numpy as np
import pickle


def loadsentspair(df_type):
    if df_type == 'train':
        df = pd.read_csv('../../dataset/stsbenchmark/sts-train.csv', delimiter='\t', error_bad_lines=False,
                         quoting=csv.QUOTE_NONE)
    elif df_type == 'dev':
        df = pd.read_csv('../../dataset/stsbenchmark/sts-dev.csv', delimiter='\t', error_bad_lines=False,
                         quoting=csv.QUOTE_NONE)
    else:
        df = pd.read_csv('../../dataset/stsbenchmark/sts-test.csv', delimiter='\t', error_bad_lines=False,
                         quoting=csv.QUOTE_NONE)
    df = df.dropna(how='any')

    sents_left = df.iloc[:, [5]].values
    sents_right = df.iloc[:, [6]].values
    score = df.iloc[:, [4]].values
    return sents_left, sents_right, score


if __name__ == "__main__":

    ser = MyBertServer()
    cli = MyBertClient()

    df_types = ['train', 'dev', 'test']

    prefix_file_name = '../../dataset/stsbenchmark/'
    for df_type in df_types:

        file_name = prefix_file_name + df_type + '.pkl'
        sents_left, sents_right, scores = loadsentspair(df_type)
        sents_left = sents_left.tolist()
        sents_right = sents_right.tolist()
        scores = scores.tolist()

        data_list = []
        for index, value in enumerate(sents_left):
            sent_left = sents_left[index][0]
            sent_right = sents_right[index][0]

            sent_left_tensor = cli.query_sentence_vec(sent_left)
            sent_right_tensor = cli.query_sentence_vec(sent_right)
            score = scores[index][0]
            data_list.append((sent_left_tensor, sent_right_tensor, score))
        with open(file_name, 'wb') as file:
            pickle.dump(data_list, file)
            print(file_name + 'has been written.')

        print('wait to terminate this programme')
