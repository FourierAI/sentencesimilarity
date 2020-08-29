#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: evaluate_model.py
# @time: 2020-07-10 19:12
# @desc:

import sys

sys.path.append('../')
from prj_model.SimaeseNet import SimaeseNet
from dataprocess.dataset import StsDataset
from prj_config.embeddingconfig import EmbeddingConfig
import torch

if __name__ == "__main__":

    embedding_config = EmbeddingConfig()
    embedding_dim = embedding_config.dimension
    dataset = StsDataset('test')
    simaese_net = SimaeseNet()
    simaese_net.load_state_dict(torch.load('simaese_net.pkl'))

    loss = []
    for left_sent, right_sent, score in dataset:
        similarity = simaese_net(left_sent.view(1, -1, embedding_dim), right_sent.view(1, -1, embedding_dim))
        print('computed similarity:{}, and score:{}'.format(similarity, score))
        diff = torch.abs(similarity - score) / 5
        loss.append(diff)

    mean_loss = sum(loss) / len(loss)
    precision = 1 - mean_loss
    print('mean loss is {}'.format(mean_loss))
    print('precision is {}'.format(precision))
