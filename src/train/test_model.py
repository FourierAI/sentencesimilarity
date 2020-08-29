#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: test_model.py
# @time: 2020-07-10 19:12
# @desc:

import sys

sys.path.append('../')
from prj_model.SimaeseNet import SimaeseNet
from dataprocess.dataset import StsDataset

if __name__ == "__main__":

    dataset = StsDataset()
    simaese_net = SimaeseNet()

    for left_sent, right_sent, score in dataset:
        similarity = simaese_net(left_sent, right_sent)
        print(similarity)
