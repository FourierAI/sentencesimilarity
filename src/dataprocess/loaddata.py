#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: loaddata.py
# @time: 2020-07-06 09:55
# @desc:

import csv

import pandas as pd
from nltk.tokenize import word_tokenize


def load_data_csv(data_set, type):
    if data_set == "sts":
        if type == "train":
            data_frame = pd.read_csv('../dataset/stsbenchmark/sts-train.csv', delimiter='\t', error_bad_lines=False,
                                     quoting=csv.QUOTE_NONE)
        elif type == "test":
            data_frame = pd.read_csv('../dataset/stsbenchmark/sts-test.csv', delimiter='\t', error_bad_lines=False,
                                     quoting=csv.QUOTE_NONE)
        elif type == 'dev':
            data_frame = pd.read_csv('../dataset/stsbenchmark/sts-dev.csv', delimiter='\t', error_bad_lines=False,
                                     quoting=csv.QUOTE_NONE)
        else:
            data_frame = None

    return data_frame.dropna(how='any')


def load_sents_pair(data_set='sts', type='train'):
    df = load_data_csv(data_set=data_set, type=type)
    sents_left = df.iloc[:, [5]].values
    sents_right = df.iloc[:, [6]].values
    return sents_left, sents_right


def load_score(data_set='sts', type='train'):
    train_df = load_data_csv(data_set=data_set, type=type)
    train_score = train_df.iloc[:, [4]].values
    return train_score


def load_sents_score(type='train'):
    df = load_data_csv('sts', type)
    sents_left = df.iloc[:, [5]].values
    sents_right = df.iloc[:, [6]].values
    score = df.iloc[:, [4]].values
    return sents_left, sents_right, score


def load_all_sents(data_set='sts'):
    train_sents_left, train_sents_right = load_sents_pair(data_set=data_set, type='train')
    test_sents_left, test_sents_right = load_sents_pair(data_set=data_set, type='test')
    dev_sents_left, dev_sents_right = load_sents_pair(data_set=data_set, type='dev')
    all_sents = []
    all_sents.extend(train_sents_left.tolist())
    all_sents.extend(train_sents_right.tolist())
    all_sents.extend(test_sents_left.tolist())
    all_sents.extend(test_sents_right.tolist())
    all_sents.extend(dev_sents_left.tolist())
    all_sents.extend(dev_sents_right.tolist())
    whole_sents = [word_tokenize(sent[0].lower()) for sent in all_sents]
    return whole_sents


if __name__ == "__main__":
    load_sents_pair()
