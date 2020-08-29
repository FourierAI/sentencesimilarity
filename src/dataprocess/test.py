#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: test.py
# @time: 2020-07-06 15:06
# @desc:
from gensim.models import Word2Vec

if __name__ == "__main__":

    model = Word2Vec.load('Word2Vec.prj_model')
    print(model.wv.__getitem__('i'))
    print(model.wv.most_similar('china'))
    print(model.wv.most_similar('like'))
