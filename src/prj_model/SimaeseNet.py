#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: DNN.py
# @time: 2020-07-23 16:18
# @desc:

import torch
import torch.nn as nn
import torch.nn.functional as F

from prj_config.embeddingconfig import EmbeddingConfig
from prj_config.global_config import GlobalConfig
import torch.nn.utils.rnn as rnn_utils


class SimaeseNet(nn.Module):

    def __init__(self):
        super(SimaeseNet, self).__init__()
        embedding_config = EmbeddingConfig()
        embedding_dimension = embedding_config.dimension

        if embedding_config.pre_train == False:
            self.rnn = nn.LSTM(
                input_size=embedding_dimension,
                hidden_size=embedding_dimension * 2,
                num_layers=1,
                batch_first=True,  # e.g. (batch, time_step, input_size)
            )
        else:
            # Default Bert dimension 768
            self.rnn = nn.LSTM(
                input_size=768,
                hidden_size=200,
                num_layers=1,
                batch_first=True,  # e.g. (batch, time_step, input_size)
            )

    def forward(self, x, y):
        x_out, (h_nx, h_cx) = self.rnn(x, None)
        y_out, (h_ny, h_cy) = self.rnn(y, None)

        BATCH_SIZE = x.shape[0]
        left_hidden = h_nx.squeeze()
        right_hidden = h_ny.squeeze()

        distance = F.pairwise_distance(left_hidden.view(BATCH_SIZE, -1), right_hidden.view(BATCH_SIZE, -1))

        cos_similarity = 5 * torch.exp(-1 * distance)

        return cos_similarity


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    dnn = SimaeseNet()

    dnn.apply(weight_init)

    test_input = torch.tensor([[1.0] * 768] * 5)

    output = dnn(test_input)

    print(output)
