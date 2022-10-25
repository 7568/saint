# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/10/25
Description:
"""
import logging
import sys


def init_log():
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    sys.stderr = open(f'log/train_v2_std_out.log', 'a')
    sys.stdout = open(f'log/train_v2_std_out.log', 'a')
    handler = logging.FileHandler(f'log/train_v2_debug_info.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

