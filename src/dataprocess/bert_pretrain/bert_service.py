#!/usr/bin/env python
# encoding: utf-8

# @author: Zhipeng Ye
# @contact: Zhipeng.ye19@xjtlu.edu.cn
# @file: bert_service.py
# @time: 2020-07-26 12:06
# @desc:
from bert_serving.server.helper import get_args_parser
from bert_serving.server import BertServer


class MyBertServer():
    def __init__(self):
        args = get_args_parser().parse_args(
            ['-model_dir', '/Data_HDD/zhipengye/projects/bert/multi_cased_L-12_H-768_A-12',
             '-port', '5555',
             '-port_out', '5556',
             '-max_seq_len', 'NONE',
             '-pooling_strategy', 'NONE',
             '-mask_cls_sep',
             '-cpu'])
        self.server = BertServer(args)
        self.server.start()
        print('bert sever has started')

    def shutdown(self):
        self.server.shutdown(port=5555)

    def start(self):
        self.server.start()
