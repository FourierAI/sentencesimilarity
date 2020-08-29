#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: train_word2vec.py
# @time: 2020-07-10 20:21
# @desc:

from dataprocess.loaddata import load_all_sents
from gensim.models import Word2Vec

if __name__ == "__main__":

    model = Word2Vec()

