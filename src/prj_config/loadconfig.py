#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: loadconfig.py
# @time: 2020-07-06 17:26
# @desc:
import yaml

def load_yaml_config(config_file_name):
    file_content = open(config_file_name)
    embedding_config = yaml.load(file_content)
    return embedding_config

if __name__ == "__main__":

    embedding_config = load_yaml_config('embedding_config.yml')

    print(embedding_config)