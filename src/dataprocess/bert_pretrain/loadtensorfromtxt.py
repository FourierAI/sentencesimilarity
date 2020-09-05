#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: loadtensorfromtxt.py
# @time: 2020-09-05 11:24
# @desc:

import pickle

if __name__ == "__main__":

    with open('../../dataset/stsbenchmark/train.pkl','rb') as file:
        train_data = pickle.load(file)
        print(train_data)