# !/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: train_model.py
# @time: 2020-07-09 20:35
# @desc:

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataprocess.dataset import StsDataset
from prj_model.SimaeseNet import SimaeseNet
from prj_config.embeddingconfig import EmbeddingConfig
from prj_config.global_config import GlobalConfig
import torch.nn.utils.rnn as rnn_utils

if __name__ == "__main__":

    global_config = GlobalConfig()
    embedding_config = EmbeddingConfig()
    EPOCH = global_config.epoch
    BATCH_SIZE = global_config.batch_size
    input_dim = embedding_config.dimension
    data_set = StsDataset('train')
    dataloader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                            collate_fn=data_set.collate_fn)

    simaese_net = SimaeseNet()

    optimizer = torch.optim.Adam(simaese_net.parameters(), lr=global_config.LEARNING_RATE)
    loss_func = nn.MSELoss()

    mean_loss_list = []
    for epoch in range(EPOCH):

        LOSS = []

        for step, (left_sent, left_len, right_sent, right_len, score) in enumerate(dataloader):

            # left_sent_pack = rnn_utils.pack_padded_sequence(left_sent, left_len, batch_first=True)
            # right_sent_pack = rnn_utils.pack_padded_sequence(right_sent, right_len, batch_first=True)

            cos_similarity = simaese_net(left_sent,
                                         right_sent)

            if step % 100 == 0:
                print(
                    'The label similarity:{} and the computation similarity:{}'.format(score, cos_similarity))

            if score.shape != cos_similarity.shape:
                print('error happen. score is {} and similarity is {}'.format(score, cos_similarity))

            loss = loss_func(cos_similarity.float(), score.float())
            LOSS.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        mean_loss = sum(LOSS) / len(LOSS)

        print('Current {} EPOCH, mean loss is {}'.format(epoch, mean_loss))
        mean_loss_list.append(mean_loss)

    torch.save(simaese_net.state_dict(), 'simaese_net.pkl')

    with open('mean_loss', 'a') as file:
        for mean_loss in mean_loss_list:
            file.write(str(mean_loss))
