#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: LSTM.py
# @time: 2020-07-09 20:23
# @desc:
from torch import nn
from prj_config.embeddingconfig import EmbeddingConfig


class RNN(nn.Module):
    def __init__(self):

        embedding_config = EmbeddingConfig()
        embedding_dimension = embedding_config.dimension

        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=embedding_dimension,
            hidden_size=embedding_dimension * 2,
            num_layers=1,
            batch_first=True,  # e.g. (batch, time_step, input_size)
        )

    def forward(self, x, y):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)
        # choose last timestep hidden h_n as output
        out = r_out[:, -1, :]
        return out


