# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/11/3
Description:
"""
# -*- coding: UTF-8 -*-
"""
Created by louis at 2022/11/3
Description:
"""
import os
import sys

sys.path.append(os.path.dirname("../../*"))
sys.path.append(os.path.dirname("../*"))
import torch.nn as nn
import torch
from einops import rearrange


class Transformer(nn.Module):
    """
    """

    def __init__(self, input_size, output_size, d_model=512, nhead=8, num_encoder_layers=3, dim_feedforward=2048,
                 dropout=0.1, activation="relu"):
        super(Transformer, self).__init__()
        self.enc = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.linear = nn.Linear(d_model * 5, output_size)

    def forward(self, x, return_fea=False, just_use_classify=False):
        if just_use_classify:
            return self.linear(x)
        x = rearrange(x, 'b s 1 n -> b s n')
        # x_2 = rearrange(x_2, 'b s 1 n -> b s n')
        x = self.enc(x)
        x = rearrange(x, 'b s n -> s b n')
        # x_2 = self.enc_1(x_2)
        memory = self.encoder(x)
        # memory_2 = self.encoder(x_2)
        memory = rearrange(memory, 's b n -> b (s n)')
        out = self.linear(memory)

        if return_fea:
            return memory, out
        else:
            return out
