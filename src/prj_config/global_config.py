#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: global_config.py
# @time: 2020-07-06 16:29
# @desc:
from prj_config.loadconfig import load_yaml_config

class GlobalConfig:
    def __init__(self):
        
        global_config = load_yaml_config('../../config_file/global_config.yml')
        self.train_or_test = global_config['train_or_test']
        self.epoch = global_config['epoch']
        self.batch_size = global_config['batch_size']
        self.data_set = global_config['data_set']
        self.LEARNING_RATE = global_config['LEARNING_RATE']

    def __str__(self):
        return str(self.__dict__)


if __name__ == "__main__":

    config = GlobalConfig()
    print(config)