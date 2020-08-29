#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: embedding_config.py
# @time: 2020-07-06 15:58
# @desc:
from prj_config.loadconfig import load_yaml_config

class EmbeddingConfig:

    def __init__(self):

        embedding_config = load_yaml_config('../../config_file/embedding_config.yml')

        self.dimension = embedding_config['dimension']
        self.pre_train = embedding_config['pre_train']
        self.pre_train_model = embedding_config['pre_train_model']
        # 0:CBOW 1:skip-gram
        self.Word2Vec_model = embedding_config['Word2Vec_model']
        self.Word2Vec_WindowSize = embedding_config['Word2Vec_WindowSize']
        self.Word2Vec_Worker = embedding_config['Word2Vec_Worker']
        self.Word2Vec_minicount = embedding_config['Word2Vec_minicount']

    def __str__(self):
        return str(self.__dict__)


if __name__ =="__main__":

    config = EmbeddingConfig()

    print(config)