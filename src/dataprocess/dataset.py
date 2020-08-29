#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: dataset.py
# @time: 2020-08-16 00:05
# @desc:

from torch.utils.data import Dataset
import pandas as pd
import csv
import dataprocess.loaddata as ld
import torch
from dataprocess.embeddingword import EmbeddingConfig
import pickle
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils


class StsDataset(Dataset):

    def __init__(self, dataset_type):
        embedding_config = EmbeddingConfig()

        self.samples = []

        if dataset_type == "train":
            data_frame = pd.read_csv('../dataset/stsbenchmark/sts-train.csv', delimiter='\t', error_bad_lines=False,
                                     quoting=csv.QUOTE_NONE)
        elif dataset_type == 'test':
            data_frame = pd.read_csv('../dataset/stsbenchmark/sts-test.csv', delimiter='\t', error_bad_lines=False,
                                     quoting=csv.QUOTE_NONE)
        elif dataset_type == 'dev':
            data_frame = pd.read_csv('../dataset/stsbenchmark/sts-dev.csv', delimiter='\t', error_bad_lines=False,
                                     quoting=csv.QUOTE_NONE)

        sents_left, sents_right, score = self.load_sents_pair(data_frame)

        if dataset_type == 'train':
            whole_sents = ld.load_all_sents()
            word_set, word_dict = self.get_wordset_and_worddict(whole_sents)

            word_capability = len(word_set)
            word_embedding = torch.nn.Embedding(word_capability, embedding_config.dimension)
            with open('word_dict.model', 'wb') as file:
                pickle.dump(word_dict, file)
            with open('word_embedding.model', 'wb') as file:
                pickle.dump(word_embedding, file)
        else:
            with open('word_dict.model', 'rb') as file:
                word_dict = pickle.load(file)

            with open('word_embedding.model', 'rb') as file:
                word_embedding = pickle.load(file)

        left_sents_embedding = self.sent2tensor(sents_left, word_embedding, word_dict)
        right_sents_embedding = self.sent2tensor(sents_right, word_embedding, word_dict)
        score = score[:, 0].tolist()

        for index, value in enumerate(score):
            self.samples.append((left_sents_embedding[index], right_sents_embedding[index], score[index]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):

        return self.samples[item]

    def get_wordset_and_worddict(self, whole_sents):
        word_set = set()
        word_dict = {}
        index = 0
        for sent in whole_sents:
            for word in sent:
                if word not in word_set:
                    word_dict[word] = index
                    word_set.add(word)
                    index += 1
        return word_set, word_dict

    def sent2tensor(self, sentences, word_embedding, word_dict):
        sents_embedding = []
        sents = [word_tokenize(sent[0].lower()) for sent in sentences]
        for sent in sents:
            sent_embedding = []
            for word in sent:
                sent_embedding.append(word_embedding(torch.LongTensor([word_dict[word]])).tolist()[0])
            sents_embedding.append(torch.tensor(sent_embedding))
        return sents_embedding

    def load_sents_pair(self, data_frame):
        score = data_frame.iloc[:, [4]].values
        sents_left = data_frame.iloc[:, [5]].values
        sents_right = data_frame.iloc[:, [6]].values
        return sents_left, sents_right, score

    def collate_fn(self, data):
        data.sort(key=lambda x: len(x[0]), reverse=True)
        left_sent_len = [len(sq[0]) for sq in data]
        right_sent_len = [len(sq[1]) for sq in data]

        left_sent = [sq[0] for sq in data]
        right_sent = [sq[1] for sq in data]
        pair_score = [sq[2] for sq in data]

        left_sent_padding = rnn_utils.pad_sequence(left_sent, batch_first=True, padding_value=0)
        right_sent_padding = rnn_utils.pad_sequence(right_sent, batch_first=True, padding_value=0)

        return left_sent_padding, left_sent_len, right_sent_padding, right_sent_len, torch.tensor(pair_score)


if __name__ == "__main__":
    dataset = StsDataset('test')
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn)

    # for step, batch in enumerate(dataloader):
    #     left_sent, left_len, right_sent, right_len, score = batch
    #
    #     print('score', score)

    for value in dataset:
        print(value)